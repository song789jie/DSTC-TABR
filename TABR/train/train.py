import argparse
import math

import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
import logging
from model import Actor, Critic
from test import valid, test
import env
import load_trace
from replay_memory import ReplayMemory

S_INFO = 6
S_LEN = 8
A_DIM = 4
LEARNING_RATE_ACTOR = 0.00005
LEARNING_RATE_CRITIC = 0.0005
TACTILE_CHUNCK_LEN_64 = 64 / 2800

TACTILE_BIT_RATE = [[2.51976, 3.59517, 4.86501, 8.08464],
                    [6.71976, 7.79517, 9.06501, 11.28464],
                    [9.38751, 10.20676, 11.2304, 14.91361],
                    [12.39386, 13.23207, 14.26618, 16.92485]]
M_IN_K = 1000.0
DEFAULT_BIT_RATE_LEVEL = 0
DEFAULT_DELAY_LEVEL = 0
UPDATE_INTERVAL = 20
MILLISECONDS_IN_SECOND = 1000.0
LATENCY_LIMIT = 45 / MILLISECONDS_IN_SECOND
LANTENCY_PENALTY = 10
REBUF_PENALTY = 1.85
SKIP_PENALTY = 1.85
SMOOTH_PENALTY = 0.2

SUMMARY_DIR = '../Results/sim/'
LOG_FILE = '../Results/sim/'

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='Evaluate only')

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


def train():
    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)

    with open(LOG_FILE + '_record', 'w') as log_file, open(LOG_FILE + '_test', 'w') as test_log_file:

        all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace()

        model_actor = Actor(A_DIM).type(dtype)
        model_critic = Critic(A_DIM).type(dtype)

        model_actor.train()
        model_critic.train()

        optimizer_actor = optim.RMSprop(model_actor.parameters(), lr=LEARNING_RATE_ACTOR)
        optimizer_critic = optim.RMSprop(model_critic.parameters(), lr=LEARNING_RATE_CRITIC)

        state = np.zeros((S_INFO, S_LEN))
        state = torch.from_numpy(state)
        last_a_bit_rate = DEFAULT_BIT_RATE_LEVEL
        last_a_delay = DEFAULT_DELAY_LEVEL

        epoch = 0

        agent_num = 4
        episode_steps = 20

        gamma = 0.99
        ent_coeff = 10
        memory = ReplayMemory(235)
        env_abr = [env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw) for i in
                   range(agent_num)]

        state_ini = [state for i in range(agent_num)]
        last_bit_rate_ini = [last_a_bit_rate for i in range(agent_num)]
        last_delay_ini = [last_a_delay for i in range(agent_num)]
        count = 0
        episode = 0
        while True:
            for agent in range(agent_num):
                states = []
                actions_bit = []
                actions_delay = []
                rewards_comparison = []
                rewards = []
                values = []
                returns = []
                advantages = []

                state = state_ini[agent]
                last_a_bit_rate = last_bit_rate_ini[agent]
                last_a_delay = last_delay_ini[agent]

                for step in range(episode_steps):
                    a_bit_rate = last_a_bit_rate
                    a_delay = last_a_delay
                    i = 0
                    reward_all = 0
                    reward_all_A = 0
                    reward_all_B = 0

                    send_data_size_all = 0
                    rebuf_all = 0
                    buffer_size_all = 0
                    download_all = 0
                    end_delay_all = 0
                    skip_frame_time_len_all = 0
                    while True:

                        time, time_interval, send_data_size, chunk_len, \
                            rebuf, buffer_size, play_time_len, end_delay, \
                            cdn_newest_id, download_id, cdn_has_frame, skip_frame_time_len, decision_flag, \
                            buffer_flag, cdn_flag, skip_flag, end_of_tactile = \
                            env_abr[agent].get_tactile_frame(a_bit_rate,
                                                             a_delay)

                        rebuf_all += rebuf
                        skip_frame_time_len_all += skip_frame_time_len

                        if end_delay >= LATENCY_LIMIT:
                            MEET_DEAD_LINE = 0
                        else:
                            MEET_DEAD_LINE = 1

                        if not cdn_flag:
                            count = count + 1
                            send_data_size_all += send_data_size

                            if TACTILE_BIT_RATE[a_delay][a_bit_rate] <= TACTILE_BIT_RATE[2][0]:
                                bit_rate = TACTILE_BIT_RATE[a_delay][a_bit_rate]
                            else:
                                bit_rate = TACTILE_BIT_RATE[2][0] + 10 * math.log10(
                                    TACTILE_BIT_RATE[a_delay][a_bit_rate] / TACTILE_BIT_RATE[2][0])

                            reward_frame = (
                                    bit_rate * play_time_len
                                    * MEET_DEAD_LINE
                                    - REBUF_PENALTY * rebuf
                                    - SKIP_PENALTY * skip_frame_time_len)

                            reward_frame_A = bit_rate * play_time_len * MEET_DEAD_LINE
                            reward_frame_B = REBUF_PENALTY * rebuf + SKIP_PENALTY * skip_frame_time_len

                            buffer_size_all += buffer_size
                            end_delay_all += end_delay
                            download_all += time_interval
                            i = i + 1

                        else:
                            if TACTILE_BIT_RATE[a_delay][a_bit_rate] <= TACTILE_BIT_RATE[2][0]:
                                bit_rate = TACTILE_BIT_RATE[a_delay][a_bit_rate]
                            else:
                                bit_rate = TACTILE_BIT_RATE[2][0] + 2 * math.log10(
                                    TACTILE_BIT_RATE[a_delay][a_bit_rate] / TACTILE_BIT_RATE[2][0])

                            reward_frame = (bit_rate * play_time_len
                                            * MEET_DEAD_LINE
                                            - REBUF_PENALTY * rebuf
                                            - SKIP_PENALTY * skip_frame_time_len)

                            reward_frame_A = bit_rate * play_time_len * MEET_DEAD_LINE
                            reward_frame_B = REBUF_PENALTY * rebuf + SKIP_PENALTY * skip_frame_time_len

                        if decision_flag or end_of_tactile:
                            if TACTILE_BIT_RATE[a_delay][a_bit_rate] <= TACTILE_BIT_RATE[2][0]:
                                bit_rate = TACTILE_BIT_RATE[a_delay][a_bit_rate]
                            else:
                                bit_rate = TACTILE_BIT_RATE[2][0] + 2 * math.log10(
                                    TACTILE_BIT_RATE[a_delay][a_bit_rate] / TACTILE_BIT_RATE[2][0])

                            if TACTILE_BIT_RATE[last_a_delay][last_a_bit_rate] <= TACTILE_BIT_RATE[2][0]:
                                last_bit_rate = TACTILE_BIT_RATE[last_a_delay][last_a_bit_rate]
                            else:
                                last_bit_rate = TACTILE_BIT_RATE[2][0] + 2 * math.log10(
                                    TACTILE_BIT_RATE[last_a_delay][last_a_bit_rate] / TACTILE_BIT_RATE[2][0])

                            reward_all += -1 * SMOOTH_PENALTY * (
                                abs(bit_rate - last_bit_rate))

                            reward_all += - LANTENCY_PENALTY * end_delay_all / count
                            reward_all_C = LANTENCY_PENALTY * end_delay_all / count
                            reward_all_D = (SMOOTH_PENALTY * (abs(bit_rate - last_bit_rate)))
                            last_a_bit_rate = a_bit_rate
                            last_a_delay = a_delay

                            state = np.roll(state, -1, axis=1)
                            state[0, -1] = TACTILE_BIT_RATE[a_delay][a_bit_rate] / float(TACTILE_BIT_RATE[-1][-1])
                            state[1, -1] = float(end_delay_all / count / LATENCY_LIMIT)
                            state[2, -1] = float(send_data_size_all / count /
                                                 TACTILE_BIT_RATE[-1][-1] * TACTILE_CHUNCK_LEN_64)
                            state[3, -1] = float(download_all / count / LATENCY_LIMIT)
                            state[4, -1] = float(buffer_size_all / count / LATENCY_LIMIT)
                            state[5, -1] = float((skip_frame_time_len_all + rebuf_all) / count / LATENCY_LIMIT)

                            state = torch.from_numpy(state)
                            send_data_size_all = 0
                            rebuf_all = 0
                            buffer_size_all = 0
                            end_delay_all = 0
                            skip_frame_time_len_all = 0

                            prob_bit, prob_delay = model_actor(state.unsqueeze(0).type(dtype))
                            action_bit = prob_bit.multinomial(num_samples=1).detach()
                            action_delay = prob_delay.multinomial(num_samples=1).detach()

                            v = model_critic(state.unsqueeze(0).type(dtype)).detach().cpu()
                            values.append(v)

                            log_file.flush()
                            rewards.append(reward_all)
                            rewards_comparison.append(torch.tensor([reward_frame]))
                            a_bit_rate = int(action_bit.squeeze().cpu().numpy())
                            a_delay = int(action_delay.squeeze().cpu().numpy())
                            actions_bit.append(torch.tensor([action_bit]))
                            actions_delay.append(torch.tensor([action_delay]))
                            states.append(state.unsqueeze(0))

                            reward_all = 0
                            reward_all_A = 0
                            reward_all_B = 0
                            reward_frame_A = 0
                            reward_frame_B = 0
                            reward_frame = 0
                            download_all = 0
                            count = 0

                        if end_of_tactile:
                            state = np.zeros((S_INFO, S_LEN))
                            state = torch.from_numpy(state)
                            last_a_bit_rate = DEFAULT_BIT_RATE_LEVEL
                            last_a_delay = DEFAULT_DELAY_LEVEL

                        reward_all += reward_frame
                        reward_all_A += reward_frame_A
                        reward_all_B += reward_frame_B

                        if len(rewards) == 235:
                            episode = episode + 1
                            state_ini[agent] = state
                            last_bit_rate_ini[agent] = last_a_bit_rate
                            last_delay_ini[agent] = last_a_delay
                            R = torch.zeros(1, 1)
                            if not end_of_tactile:
                                v = model_critic(state.unsqueeze(0).type(dtype))
                                v = v.detach().cpu()
                                R = v.data

                            values.append(Variable(R))
                            R = Variable(R)

                            for i in reversed(range(len(rewards))):
                                R = gamma * R + rewards[i]
                                returns.insert(0, R)

                                td = R - values[i]
                                advantages.insert(0, td)
                            if torch.eq(states[0][0], torch.from_numpy(np.zeros((S_INFO,
                                                                                 S_LEN)))).sum() == S_INFO * S_LEN:
                                memory.push([states[1:], actions_bit[1:], actions_delay[1:], returns[1:], advantages[1:]])
                            else:
                                memory.push([states, actions_bit, actions_delay, returns, advantages])

                            model_actor.zero_grad()
                            model_critic.zero_grad()

                            # mini_batch
                            batch_size = memory.return_size()
                            # print(batch_size)
                            batch_states, batch_actions_bit, batch_actions_delay, batch_returns, batch_advantages = memory.pop(
                                batch_size)

                            states = []
                            actions_bit = []
                            actions_delay = []
                            rewards_comparison = []
                            rewards = []
                            values = []
                            returns = []
                            advantages = []

                            probs_pre_bit, probs_pre_delay = model_actor(batch_states.type(dtype))
                            values_pre = model_critic(batch_states.type(dtype))

                            # actor_loss
                            prob_value_bit = torch.gather(probs_pre_bit, dim=1, index=batch_actions_bit.unsqueeze(1).type(dlongtype))
                            prob_value_delay = torch.gather(probs_pre_delay, dim=1,
                                                            index=batch_actions_delay.unsqueeze(1).type(dlongtype))
                            policy_loss_bit = -torch.mean(prob_value_bit * batch_advantages.type(dtype))
                            policy_loss_delay = -torch.mean(prob_value_delay * batch_advantages.type(dtype))
                            loss_ent_bit = ent_coeff * torch.mean(probs_pre_bit * torch.log(probs_pre_bit + 1e-5))
                            loss_ent_delay = ent_coeff * torch.mean(probs_pre_delay * torch.log(probs_pre_delay + 1e-5))
                            actor_loss_bit = policy_loss_bit + loss_ent_bit
                            actor_loss_delay = policy_loss_delay + loss_ent_delay

                            # critic_loss
                            vf_loss = (values_pre - batch_returns.type(dtype)) ** 2  # V_\theta - Q'

                            critic_loss = 0.5 * torch.mean(vf_loss)

                            # update
                            actor_total_loss = policy_loss_bit + loss_ent_bit + policy_loss_delay + loss_ent_delay
                            optimizer_actor.zero_grad()
                            optimizer_critic.zero_grad()
                            actor_total_loss.backward()
                            optimizer_actor.step()
                            critic_loss.backward()
                            optimizer_critic.step()
                            memory.clear()
                            logging.info('Epoch: ' + str(epoch) + str(step) +
                                         ' Avg_policy_loss: ' + str(
                                policy_loss_bit.detach().cpu().numpy() + policy_loss_delay.detach().cpu().numpy()) +
                                         ' Avg_value_loss: ' + str(critic_loss.detach().cpu().numpy()) +
                                         ' Avg_entropy_loss: ' + str(
                                A_DIM * loss_ent_bit.detach().cpu().numpy() + A_DIM * loss_ent_delay.detach().cpu().numpy()))

                            logging.info("Model saved in file")

                        if end_of_tactile:
                            break

                epoch += 1
                print(epoch, episode)
                valid(model_actor, epoch, test_log_file)
                if epoch % UPDATE_INTERVAL == 0:
                    ent_coeff = 0.95 * ent_coeff


def main():
    train()


if __name__ == '__main__':
    main()
