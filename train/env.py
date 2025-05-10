import numpy as np
import datetime
import os

BIT_RATE_LEVELS = 4
DELAY_LEVELS = 4
TACTILE_SIZE_FILE = "../dataset/tactile/bit_rate_"

MILLISECONDS_IN_SECOND = 1000.0
b_IN_Mb = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 2024
TACTILE_CHUNCK_LEN_8 = 8 / 2800
TACTILE_CHUNCK_LEN_64 = 64 / 2800

TACTILE_CODING_DELAY = [[2 / MILLISECONDS_IN_SECOND, 1 / MILLISECONDS_IN_SECOND, 0.5 / MILLISECONDS_IN_SECOND,
                         0.25 / MILLISECONDS_IN_SECOND],
                        [3 / MILLISECONDS_IN_SECOND, 2 / MILLISECONDS_IN_SECOND, 1.5 / MILLISECONDS_IN_SECOND,
                         1.25 / MILLISECONDS_IN_SECOND],
                        [3 / MILLISECONDS_IN_SECOND, 2 / MILLISECONDS_IN_SECOND, 1.5 / MILLISECONDS_IN_SECOND,
                         1.25 / MILLISECONDS_IN_SECOND],
                        [3 / MILLISECONDS_IN_SECOND, 2 / MILLISECONDS_IN_SECOND, 1.5 / MILLISECONDS_IN_SECOND,
                         1.25 / MILLISECONDS_IN_SECOND]]

TARGET_BUFFER = 64 / 2800
LATENCY_LIMIT = 45 / MILLISECONDS_IN_SECOND

DEFAULT_BIT_RATE_LEVEL = 1
DEFAULT_DELAY_LEVEL = 1


class Environment:
    def __init__(
            self,
            all_cooked_time,
            all_cooked_bw,
            random_seed=RANDOM_SEED,
            set_idx=None,
    ):
        logfile_path = "./Results/sim/env_log/"
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        os.makedirs(logfile_path, exist_ok=True)
        self.log_file = open(logfile_path + "log." + current_time, "w")

        assert len(all_cooked_time) == len(all_cooked_bw)
        # np.random.seed(random_seed)
        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw
        self.time = -2.0
        self.play_time = -2.0
        self.play_time_counter = 0
        self.newest_chunk = 0
        self.next_decision_chunk = 20

        self.skip = False
        self.tactile_chunk_counter = 0
        self.buffer_size = 0

        if set_idx is None:
            self.trace_idx = np.random.randint(len(self.all_cooked_time))
        else:
            self.trace_idx = set_idx

        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]
        self.decision = False
        self.buffer_status = True

        self.skip_time_chunk = 1000000000
        self.add_chunk = 0
        self.skip_to_chunk = self.skip_time_chunk

        self.tactile_size = {}
        self.gop_flag = {}
        self.cdn_arrive_time = {}
        self.gop_time_len = 64 / 2800
        self.latency = 2 ** (3 - DEFAULT_DELAY_LEVEL) * TACTILE_CHUNCK_LEN_8  # 等待缓冲时间

        for bitrate in range(BIT_RATE_LEVELS):
            self.tactile_size[bitrate] = {}
            self.gop_flag[bitrate] = {}
            self.cdn_arrive_time[bitrate] = {}
            for delay in range(DELAY_LEVELS):
                self.tactile_size[bitrate][delay] = []
                self.gop_flag[bitrate][delay] = []
                self.cdn_arrive_time[bitrate][delay] = []
                with open(TACTILE_SIZE_FILE + str(bitrate) + '_' + str(delay) + '.txt') as f:
                    for line in f:
                        self.tactile_size[bitrate][delay].append(float(line.split()[1]))
                        self.gop_flag[bitrate][delay].append(int(float(line.split()[2])))
                        self.cdn_arrive_time[bitrate][delay].append(float(line.split()[0]))

    def get_trace_id(self):
        return self.trace_idx

    def get_tactile_frame(self, a_bit_rate, a_delay):
        assert a_bit_rate >= 0
        assert a_bit_rate < BIT_RATE_LEVELS

        assert a_delay >= 0
        assert a_delay < DELAY_LEVELS

        self.decision = False
        self.add_chunk = 0
        tactile_chunk_size = self.tactile_size[a_bit_rate][a_delay][self.tactile_chunk_counter]

        cdn_rebuf_time = 0
        end_of_tactile = False
        duration = 0
        rebuf = 0
        self.skip = False

        if (self.time < self.cdn_arrive_time[a_bit_rate][a_delay][self.tactile_chunk_counter]
                and not end_of_tactile):
            cdn_rebuf_time = self.cdn_arrive_time[a_bit_rate][a_delay][self.tactile_chunk_counter] - self.time
            self.newest_chunk = self.tactile_chunk_counter
            duration = cdn_rebuf_time

            if not self.buffer_status:
                if self.buffer_size > cdn_rebuf_time:
                    self.buffer_size -= cdn_rebuf_time
                    self.play_time += cdn_rebuf_time
                    rebuf = 0
                    play_len = cdn_rebuf_time
                else:
                    self.play_time += self.buffer_size
                    rebuf = cdn_rebuf_time - self.buffer_size
                    play_len = self.buffer_size
                    self.buffer_size = 0
                    self.buffer_status = True

                if self.play_time_counter >= self.skip_time_chunk:
                    self.play_time_counter = self.skip_to_chunk
                    self.play_time = self.play_time_counter * TACTILE_CHUNCK_LEN_64
                    self.add_chunk = 0
                else:
                    self.play_time_counter = int((self.play_time + 2) / TACTILE_CHUNCK_LEN_64)

                # print('new - counter', (self.newest_frame - self.tactile_chunk_counter))
                self.latency = (abs(self.newest_chunk - self.tactile_chunk_counter) * TACTILE_CHUNCK_LEN_64
                                + 2 ** (3 - a_delay) * TACTILE_CHUNCK_LEN_8)
                self.time = self.cdn_arrive_time[a_bit_rate][a_delay][self.tactile_chunk_counter]

            else:
                # 否则无需任何动作
                rebuf = duration
                play_len = 0
                # play_len 用于驱动self.play_time 往前走
                self.time = self.cdn_arrive_time[a_bit_rate][a_delay][self.tactile_chunk_counter]
                # self.time 往前推进
                self.latency = ((self.newest_chunk - self.tactile_chunk_counter) * TACTILE_CHUNCK_LEN_64
                                + 2 ** (3 - a_delay) * TACTILE_CHUNCK_LEN_8)

            self.log_file.write("real_time %.4f\t" % self.time +
                                "cdn_rebuf%.4f\t" % cdn_rebuf_time +
                                "client_rebuf %.3f\t" % rebuf +
                                "download_duration %.4f\t" % duration +
                                "frame_size %.4f\t" % tactile_chunk_size +
                                "play_time_len %.4f\t" % (play_len) +
                                "download_id %d\t" % (self.tactile_chunk_counter - 1) +
                                "cdn_newest_frame %d\t" % self.newest_chunk +
                                "client_buffer %.4f\t" % self.buffer_size +
                                "play_time %.4f\t" % self.play_time +
                                "play_id %.4f\t" % self.play_time_counter +
                                "latency %.4f\t" % self.latency + "111\n")

            cdn_has_chunk = []
            for bit_rate in range(BIT_RATE_LEVELS):
                for delay in range(DELAY_LEVELS):
                    cdn_has_chunk_temp = self.tactile_size[bit_rate][delay][
                                         self.tactile_chunk_counter: self.newest_chunk]
                    cdn_has_chunk.append(cdn_has_chunk_temp)
            cdn_has_chunk.append(self.gop_flag[a_bit_rate][a_delay][self.tactile_chunk_counter:self.newest_chunk])

            return [self.time,
                    duration,
                    0,
                    0,
                    rebuf,
                    self.buffer_size,
                    play_len,
                    self.latency,
                    self.newest_chunk,
                    (self.tactile_chunk_counter - 1),
                    cdn_has_chunk,
                    self.add_chunk * TACTILE_CHUNCK_LEN_64,
                    self.decision,
                    self.buffer_status,
                    True,
                    self.skip,
                    end_of_tactile]

        else:

            the_newst_chunk = self.tactile_chunk_counter
            current_new = self.cdn_arrive_time[a_bit_rate][a_delay][the_newst_chunk]
            while current_new < self.time:
                the_newst_chunk += 1
                if int((self.time + 2) / TACTILE_CHUNCK_LEN_64) == len(self.cdn_arrive_time[a_bit_rate][a_delay]):
                    break

                if the_newst_chunk + 1 >= len(self.cdn_arrive_time[a_bit_rate][a_delay]):
                    break
                current_new = self.cdn_arrive_time[a_bit_rate][a_delay][the_newst_chunk]
            self.newest_chunk = the_newst_chunk

        if (int((self.time + 2) / 0.5) >= len(self.cooked_bw)
                or self.tactile_chunk_counter + 2 >= len(self.cdn_arrive_time[a_bit_rate][a_delay])
                or int((self.time + 2) / TACTILE_CHUNCK_LEN_64) + 2 >= len(self.cdn_arrive_time[a_bit_rate][a_delay])):
            end_of_tactile = True
        else:
            throughput = self.cooked_bw[int((self.time + 2) / 0.5)] * b_IN_Mb

            if float(tactile_chunk_size * 46 / throughput) < TACTILE_CHUNCK_LEN_64:
                duration = float((TACTILE_CODING_DELAY[a_bit_rate][a_delay]
                                  + tactile_chunk_size * 46 / 2 ** a_delay / throughput))
            else:
                if a_delay == 0:
                    duration = float((TACTILE_CODING_DELAY[a_bit_rate][a_delay]
                                      + tactile_chunk_size * 46 / throughput))
                elif a_delay == 1:
                    f_duration = (3 * (tactile_chunk_size * 46 / throughput / 2) - TACTILE_CHUNCK_LEN_8 * 4) / 2
                    duration = float(TACTILE_CODING_DELAY[a_bit_rate][a_delay] + f_duration)
                elif a_delay == 2:
                    f_duration = (10 * (tactile_chunk_size * 46 / throughput / 4) - TACTILE_CHUNCK_LEN_8 * 12) / 4
                    duration = float(TACTILE_CODING_DELAY[a_bit_rate][a_delay] + f_duration)
                elif a_delay == 3:
                    f_duration = (36 * (tactile_chunk_size * 46 / throughput / 8) - TACTILE_CHUNCK_LEN_8 * 28) / 8
                    duration = float(TACTILE_CODING_DELAY[a_bit_rate][a_delay] + f_duration)

        if self.gop_flag[a_bit_rate][a_delay][self.tactile_chunk_counter + 1] == 1:
            self.decision = True
            self.next_decision_chunk = self.tactile_chunk_counter + 20 + 1

        if not end_of_tactile:
            if self.buffer_size > duration:
                self.buffer_size -= duration
                self.play_time += duration
                play_time_len = duration
                rebuf = 0
            else:
                self.play_time += self.buffer_size
                play_time_len = self.buffer_size
                rebuf = duration - self.buffer_size
                self.buffer_size = 0
                self.buffer_status = True
            if self.play_time_counter >= self.skip_time_chunk:
                self.play_time_counter = self.skip_to_chunk
                self.play_time = self.play_time_counter * TACTILE_CHUNCK_LEN_64
                self.log_file.write("ADD_Chunk" + str(self.add_chunk) + "\n")
                self.add_chunk = 0
            else:
                self.play_time_counter = int((self.play_time + 2) / TACTILE_CHUNCK_LEN_64)

            self.latency = (abs(self.newest_chunk - self.tactile_chunk_counter) * TACTILE_CHUNCK_LEN_64
                            + 2 ** (3 - a_delay) * TACTILE_CHUNCK_LEN_8
                            + duration)

            self.buffer_size += TACTILE_CHUNCK_LEN_64
            self.buffer_status = False
            self.time += duration
            if self.latency > LATENCY_LIMIT:

                self.skip_time_chunk = self.tactile_chunk_counter
                if self.newest_chunk + 1 >= self.next_decision_chunk:
                    self.add_chunk = self.next_decision_chunk - self.skip_time_chunk - 1
                    self.tactile_chunk_counter = self.next_decision_chunk
                    self.skip_to_chunk = self.tactile_chunk_counter
                    self.next_decision_chunk += 20
                    self.latency = (abs(self.newest_chunk - self.tactile_chunk_counter) * TACTILE_CHUNCK_LEN_64
                                    + 2 ** (3 - a_delay) * TACTILE_CHUNCK_LEN_8)
                    self.skip = True
                    self.decision = True
                else:
                    self.add_chunk = 0
                    self.tactile_chunk_counter += 1
                    self.skip_to_chunk = self.tactile_chunk_counter

                self.log_file.write("skip events: skip_time_frame, play_frame, new_download_frame, ADD_frame" + str(
                    self.skip_time_chunk) + " " + str(self.play_time_counter) + " " + str(
                    self.tactile_chunk_counter) + " " + str(self.add_chunk) + " " +
                                    str((self.newest_chunk - self.tactile_chunk_counter))
                                    + " " + str(self.buffer_size) + " "
                                    + str(duration > 2 ** (3 - a_delay) * TACTILE_CHUNCK_LEN_8) + "\n")

            else:
                self.tactile_chunk_counter += 1

            self.log_file.write("real_time %.4f\t" % self.time +
                                "cdn_rebuf%.4f\t" % cdn_rebuf_time +
                                "client_rebuf %.3f\t" % rebuf +
                                "download_duration %.4f\t" % duration +
                                "frame_size %.4f\t" % tactile_chunk_size +
                                "play_time_len %.4f\t" % duration +
                                "download_id %d\t" % (self.tactile_chunk_counter - 1) +
                                "cdn_newest_frame %d\t" % self.newest_chunk +
                                "client_buffer %.4f\t" % self.buffer_size +
                                "play_time %.4f\t" % self.play_time +
                                "play_id %.4f\t" % self.play_time_counter +
                                "latency %.4f\t" % self.latency + "333\n")

            cdn_has_chunk = []
            for bit_rate in range(BIT_RATE_LEVELS):
                for delay in range(DELAY_LEVELS):
                    cdn_has_chunk_temp = self.tactile_size[bit_rate][delay][
                                         self.tactile_chunk_counter: self.newest_chunk]
                    cdn_has_chunk.append(cdn_has_chunk_temp)
            cdn_has_chunk.append(self.gop_flag[a_bit_rate][a_delay][self.tactile_chunk_counter:self.newest_chunk])
            return [self.time,
                    duration,
                    tactile_chunk_size,
                    TACTILE_CHUNCK_LEN_64,
                    rebuf,
                    self.buffer_size,
                    play_time_len,
                    self.latency,
                    self.newest_chunk,
                    (self.tactile_chunk_counter - 1),
                    cdn_has_chunk,
                    self.add_chunk * TACTILE_CHUNCK_LEN_64,
                    self.decision,
                    self.buffer_status,
                    False,
                    self.skip,
                    end_of_tactile]

        # tactile 结束时
        if end_of_tactile:
            self.time = -2
            self.play_time = -2
            self.play_time_counter = 0
            self.newest_chunk = 0
            self.tactile_chunk_counter = 0
            self.buffer_size = 0

            self.trace_idx += 1
            if self.trace_idx >= len(self.all_cooked_time):
                self.trace_idx = 0

            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            self.decision = False
            self.buffer_status = True
            self.skip_time_chunk = 1000000000
            self.next_decision_chunk = 22
            self.add_chunk = 0
            self.skip_to_chunk = self.skip_time_chunk

            self.tactile_size = {}
            self.gop_flag = {}
            self.cdn_arrive_time = {}
            self.gop_time_len = 64 / 2800
            for bitrate in range(BIT_RATE_LEVELS):
                self.tactile_size[bitrate] = {}
                self.gop_flag[bitrate] = {}
                self.cdn_arrive_time[bitrate] = {}
                for delay in range(DELAY_LEVELS):
                    self.tactile_size[bitrate][delay] = []
                    self.gop_flag[bitrate][delay] = []
                    self.cdn_arrive_time[bitrate][delay] = []
                    with open(TACTILE_SIZE_FILE + str(bitrate) + '_' + str(delay) + '.txt') as f:
                        for line in f:
                            self.tactile_size[bitrate][delay].append(float(line.split()[1]))
                            self.gop_flag[bitrate][delay].append(int(float(line.split()[2])))
                            self.cdn_arrive_time[bitrate][delay].append(float(line.split()[0]))

            self.latency = 2 ** (3 - a_delay) * TACTILE_CHUNCK_LEN_8
            cdn_has_chunk = []
            for bit_rate in range(BIT_RATE_LEVELS):
                for delay in range(DELAY_LEVELS):
                    cdn_has_chunk_temp = self.tactile_size[bit_rate][delay][
                                         self.tactile_chunk_counter: self.newest_chunk]
                    cdn_has_chunk.append(cdn_has_chunk_temp)
            cdn_has_chunk.append(self.gop_flag[a_bit_rate][a_delay][self.tactile_chunk_counter:self.newest_chunk])

            return [self.time,
                    duration,
                    tactile_chunk_size,
                    TACTILE_CHUNCK_LEN_64,
                    rebuf,
                    self.buffer_size,
                    duration,
                    self.latency,
                    self.newest_chunk,
                    (self.tactile_chunk_counter - 1),
                    cdn_has_chunk,
                    0,
                    self.decision,
                    self.buffer_status,
                    False,
                    False,
                    True]
