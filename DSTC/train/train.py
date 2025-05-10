import torch
from torch import nn
from pathlib import Path
from accelerate import Accelerator
from ema_pytorch import EMA
from model.ASTCv3 import ASTC_model
from shutil import rmtree
from src.dataloader import Haptic_DataLoader, Haptic_Dataset
from src.optimizer import get_optimizer
from src.args import parse_args
from src.utils import cycle, yes_or_no, noop, accum_log, exists, checkpoint_num_steps
import numpy as np
import torch.nn.functional as F

args = parse_args()


class ASTC_Trainer(nn.Module):
    def __init__(self, model: ASTC_model,
                 *,
                 result_folder: str = './results_P4',
                 use_ema: bool = True):
        super().__init__()
        self.model = model
        self.results_folder = Path(result_folder)
        self.use_ema = use_ema
        self.accelerator = Accelerator(log_with="tensorboard", project_dir="./tensorboard_P4")

        if self.use_ema:
            self.ema_haptic_stream = EMA(self.model,
                                         beta=args.ema_beta,
                                         update_after_step=args.ema_update_after_step,
                                         update_every=args.ema_update_every)

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = args.num_train_steps
        self.batch_size = args.batch_size
        self.grad_accum_every = args.grad_accum_every
        self.lr = args.lr
        self.wd = args.weight_decay
        self.max_grad_norm = args.max_grad_norm
        self.discr_max_grad_norm = args.discr_max_grad_norm
        self.target_sample_hz = args.target_sample_hz
        self.data_length = args.data_length
        self.save_model_every = args.save_model_every
        self.save_results_every = args.save_results_every
        self.log_losses_every = args.log_losses_every
        self.apply_grad_penalty_every = args.apply_grad_penalty_every

        haptic_dataset = Haptic_Dataset(root_path=args.root_path,
                                        data_path=args.data_path,
                                        json_name=args.json_name)
        val_haptic_dataset = Haptic_Dataset(root_path=args.root_path,
                                            data_path=args.val_data_path,
                                            json_name=args.val_json_name)

        hyperparameters = {
            "num_train_steps": self.num_train_steps,
            "batch_size": self.batch_size,
            "gradient_accum_every": self.grad_accum_every,
            "learning_rate": self.lr,
            "target_sample_hz": self.target_sample_hz,
            "data_length": self.data_length
        }

        self.optim = get_optimizer(self.unwrapped_ASTC_stream.non_discr_parameters(), lr=self.lr, wd=self.wd)

        for discr_optimizer_key, discr in self.multiscale_discriminator_iter():
            one_multiscale_discr_optimizer = get_optimizer(discr.parameters(), lr=self.lr, wd=self.lr)
            one_multiscale_discr_optimizer = self.accelerator.prepare(one_multiscale_discr_optimizer)
            setattr(self, discr_optimizer_key, one_multiscale_discr_optimizer)

        self.train_dataLoader, self.valid_dataLoader = (
            Haptic_DataLoader(haptic_dataset,
                              batch_size=args.batch_size,
                              vali_dataset=val_haptic_dataset,
                              num_workers=args.num_workers).get_dataloader())

        self.train_dataLoader_iter = cycle(self.train_dataLoader)

        self.valid_dataLoader_iter = cycle(self.valid_dataLoader)

        [self.haptic_stream, self.optim, self.train_dataLoader] = \
            self.accelerator.prepare([self.model, self.optim, self.train_dataLoader])
        self.haptic_stream.to(device=self.accelerator.device)

        self.results_folder = Path(result_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no(
                'do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.accelerator.init_trackers("ASTC_model", config=hyperparameters)

    @property
    def unwrapped_ASTC_stream(self):
        return self.accelerator.unwrap_model(self.model)

    def multiscale_discriminator_iter(self):
        for ind, discr in enumerate(self.unwrapped_ASTC_stream.discriminators):
            yield f'multiscale_discr_optimizer_{ind}', discr

    def train_start(self, log_fn=noop):

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        print('training complete')
        self.accelerator.end_training()

    def train_step(self):
        device = self.accelerator.device
        self.unwrapped_ASTC_stream.to(device)
        self.ema_haptic_stream.ema_model.to(device)
        steps = int(self.steps.item())

        apply_grad_penalty = self.apply_grad_penalty_every > 0 and not (steps % self.apply_grad_penalty_every)
        log_losses = self.log_losses_every > 0 and not (steps % self.log_losses_every)

        self.haptic_stream.train()
        logs = {}

        for _ in range(self.grad_accum_every):
            haptic_wave = next(self.train_dataLoader_iter)
            haptic_wave = haptic_wave.float().to(device)
            loss, (recon_loss, adversarial_loss, feature_loss, SSIM_loss,
                   recon_loss_64, recon_loss_32, recon_loss_16, recon_loss_8,
                   feature_intermediate_loss) = \
                self.haptic_stream(haptic_wave, return_loss_breakdown=True)

            self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, dict(
                loss=loss.item() / self.grad_accum_every,
                recon_loss=recon_loss.item() / self.grad_accum_every,
                feature_intermediate_loss=feature_intermediate_loss.item() / self.grad_accum_every,
                recon_loss_64=recon_loss_64.item() / self.grad_accum_every,
                recon_loss_32=recon_loss_32.item() / self.grad_accum_every,
                recon_loss_16=recon_loss_16.item() / self.grad_accum_every,
                recon_loss_8=recon_loss_8.item() / self.grad_accum_every,
            ))

            if log_losses:
                accum_log(logs, dict(
                    adversarial_loss=adversarial_loss.item() / self.grad_accum_every,
                    feature_loss=feature_loss.item() / self.grad_accum_every,
                    SSIM_loss=SSIM_loss.item() / self.grad_accum_every,
                ))

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.haptic_stream.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        for name, multiscale_discr_optim in self.multiscale_discriminator_iter():
            multiscale_discr_optim = getattr(self, name)
            multiscale_discr_optim.zero_grad()

        for _ in range(self.grad_accum_every):
            haptic_wave = next(self.train_dataLoader_iter)
            haptic_wave = haptic_wave.float().to(device)

            discr_losses = self.haptic_stream(
                haptic_wave,
                apply_grad_penalty=apply_grad_penalty,
                return_discr_loss=True,
                return_discr_losses_separately=True
            )

            for name, discr_loss in discr_losses:
                self.accelerator.backward(discr_loss / self.grad_accum_every, retain_graph=True)
                accum_log(logs, {name: discr_loss.item() / self.grad_accum_every})

        for name, multiscale_discr_optim in self.multiscale_discriminator_iter():
            multiscale_discr_optim = getattr(self, name)
            multiscale_discr_optim.step()

        losses_str = f"{steps}: haptic_stream total loss: {logs['loss']:.3f}," \
                     f"' recon loss: {logs['recon_loss']:.3f}" \
                     f"' feature_intermediate_loss: {logs['feature_intermediate_loss']:.3f}" \
                     f"' recon loss 64: {logs['recon_loss_64']:.3f}" \
                     f"' recon loss 32: {logs['recon_loss_32']:.3f}" \
                     f"' recon loss 16: {logs['recon_loss_16']:.3f}" \
                     f"' recon loss 8: {logs['recon_loss_8']:.3f}" \
                     f"' SSIM loss: {logs['SSIM_loss']:.3f}" \

        if log_losses:
            self.accelerator.log({
                "total_loss": logs['loss'],
                "recon_loss": logs['recon_loss'],
                "adversarial_loss": logs['adversarial_loss'],
                "feature_loss": logs['feature_loss'],
            }, step=steps)

        for key, loss in logs.items():

            if not key.startswith('scale:'):
                continue
            _, scale_factor = key.split(':')

            losses_str += f" | discr (scale {scale_factor}) loss: {loss:.3f}"
            if log_losses:
                self.accelerator.log({f"discr_loss (scale {scale_factor})": loss}, step=steps)

        # log
        print(losses_str)

        self.accelerator.wait_for_everyone()

        self.ema_haptic_stream.update()

        recon_loss_8, recon_loss_16, recon_loss_32, recon_loss_64 = 0, 0, 0, 0
        self.accelerator.wait_for_everyone()

        if not (steps % self.save_results_every):
            models = [(self.unwrapped_ASTC_stream, str(steps))]
            if self.use_ema:
                models.append((self.ema_haptic_stream.ema_model if self.use_ema else self.unwrapped_haptic_stream,
                               f'{steps}.ema'))

            haptic_wave = next(self.valid_dataLoader_iter)
            haptic_wave = haptic_wave.float().to(device)

            for model, label in models:
                model.eval()

                with (torch.inference_mode()):
                    recons_8, recons_16, recons_32, recons_64 = model(
                        haptic_wave, return_recons_only=True)

                for ind, recon in enumerate(recons_8.unbind(dim=0)):
                    recon_loss_8 = F.mse_loss(haptic_wave, recons_8)
                    recon_loss_16 = F.mse_loss(haptic_wave, recons_16)
                    recon_loss_32 = F.mse_loss(haptic_wave, recons_32)
                    recon_loss_64 = F.mse_loss(haptic_wave, recons_64)
                    filename_recon = str(self.results_folder / f'sample_{label}_{ind}.txt')
                    filename_real = str(self.results_folder / f'sample_real_{label}_{ind}.txt')
                    recon_np = recon.cpu().detach().numpy()
                    haptic_wave_save = haptic_wave[ind, :].cpu().detach().numpy()
                    # np.savetxt(filename_recon, recon_np, fmt='%f', delimiter='\t')
                    # np.savetxt(filename_real, haptic_wave_save, fmt='%f', delimiter='\t')

            print(f'{steps}: saving to {str(self.results_folder)}, '
                  f'recon_loss {recon_loss_8, recon_loss_16, recon_loss_32, recon_loss_64}')

        self.accelerator.wait_for_everyone()

        if not (steps % self.save_model_every):
            model_path = str(self.results_folder / f'haptic_stream.{steps}.pt')
            self.save(model_path)

            print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    def save(self, path):
        pkg = dict(
            model=self.accelerator.get_state_dict(self.haptic_stream),
            optim=self.optim.state_dict(),
            config=self.unwrapped_ASTC_stream._configs,
        )

        if self.use_ema:
            pkg['ema_model'] = self.ema_haptic_stream.state_dict()

        for key, _ in self.multiscale_discriminator_iter():
            discr_optim = getattr(self, key)
            pkg[key] = discr_optim.state_dict()

        torch.save(self.haptic_stream, path[:-3] + 'model' + '.pt')
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location='cpu')

        self.unwrapped_haptic_stream.load_state_dict(pkg['model'])

        if self.use_ema:
            assert 'ema_model' in pkg
            self.ema_haptic_stream.load_state_dict(pkg['ema_model'])

        self.optim.load_state_dict(pkg['optim'])

        for key, _ in self.multiscale_discriminator_iter():
            discr_optim = getattr(self, key)
            discr_optim.load_state_dict(pkg[key])

        # + 1 to start from the next step and avoid overwriting the last checkpoint

        self.steps = torch.tensor([checkpoint_num_steps(path) + 1], device=self.accelerator.device)


if __name__ == '__main__':
    trainer = ASTC_Trainer(ASTC_model())
    trainer.train_start()
