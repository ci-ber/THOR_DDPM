from core.Trainer import Trainer
from time import time
from torch.optim import AdamW
import os
import wandb
import logging
from net_utils.simplex_noise import generate_noise, generate_simplex_noise
from optim.losses.image_losses import *
from torch.cuda.amp import GradScaler, autocast
from model_zoo.guided_diffusion.script_util import *
from model_zoo.guided_diffusion.respace import *
from model_zoo.guided_diffusion.resample import *
from model_zoo.guided_diffusion.gaussian_diffusion import *
from model_zoo.guided_diffusion.unet import UNetModel
from model_zoo.guided_diffusion.fp16_util import MixedPrecisionTrainer


"""
Implementation of the abstract class "Trainer" from core module.
Includes training and validation routines
"""
class PTrainer(Trainer):
    def __init__(self, training_params, model, data, device, log_wandb=True):
        super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)

        self.val_interval = training_params['val_interval']
        self.schedule_sampler = training_params['schedule_sampler'] if 'schedule_sampler' in training_params else \
            'uniform'
        self.lr = training_params['lr'] if 'lr' in training_params else 1e-4
        self.weight_decay = training_params['optimizer_params']['weight_decay'] if 'weight_decay' in training_params[
            'optimizer_params'] else 0
        self.use_fp16 = False,
        self.fp16_scale_growth = 1e-3
        self.mp_trainer = MixedPrecisionTrainer(model=model, use_fp16=self.use_fp16, fp16_scale_growth=self.fp16_scale_growth)
        self.diffusion = create_gaussian_diffusion(steps=1000)
        self.scheduler = create_named_schedule_sampler(self.schedule_sampler, self.diffusion, maxt=1000)
        self.optimizer = AdamW(self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay)


    def train(self, model_state=None, opt_state=None, start_epoch=0):
        """
        Train local client
        :param model_state: weights
            weights of the global model
        :param opt_state: state
            state of the optimizer
        :param start_epoch: int
            start epoch
        :return:
            self.model.state_dict():
        """
        if model_state is not None:
            self.model.load_state_dict(model_state)  # load weights
            self.mp_trainer = MixedPrecisionTrainer(model=self.model, use_fp16=self.use_fp16, fp16_scale_growth=self.fp16_scale_growth)
            print(f'**** WEIGHTS LOADED IN TRAINING ****')
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)  # load optimizer
            print(f'**** Optimizer LOADED IN TRAINING ****')


        epoch_losses = []
        epoch_losses_mse = []

        self.early_stop = False
        # to handle loss with mixed precision training

        for epoch in range(self.training_params['nr_epochs']):
            if start_epoch > epoch:
                continue
            if self.early_stop is True:
                logging.info("[Trainer::test]: ################ Finished training (early stopping) ################")
                break
            start_time = time()
            batch_loss, batch_loss_mse, count_images = 0, 0, 0

            for data in self.train_ds:
                # Input
                images = data[0].to(self.device)
                count_images += images.shape[0]
                transformed_images = self.transform(images) if self.transform is not None else images
                transformed_images *= 2
                transformed_images -= 1
                
                self.mp_trainer.zero_grad()
                
                # for mixed precision training
                with autocast(enabled=True):
                    # Create timesteps

                    t, weights = self.scheduler.sample(transformed_images.shape[0], self.device)
                    # print(f'************ {t} ****************')
                    losses1 = self.diffusion.training_losses(self.model, transformed_images, t)

                    losses = losses1[0]
                    sample = losses1[1]

                    loss = (losses["loss"] * weights).mean()
                    lossmse = (losses["mse"] * weights).mean().detach()

                self.mp_trainer.backward(loss)
                self.mp_trainer.optimize(self.optimizer)

                batch_loss += loss.item() * images.size(0)
                batch_loss_mse += lossmse.item() * images.size(0)

                # with torch.no_grad():
            
                    # b, c, h, w = transformed_images.shape

                    # sample, x_noisy, orig = self.diffusion.ddim_sample_loop_known(self.model,
                                                                                            #  transformed_images.shape, transformed_images)

                rec = sample.detach().cpu()[0].numpy()
                img = transformed_images.detach().cpu()[0].numpy()
                grid_image = np.hstack([img, rec])

                wandb.log({'Train' + '/Example_': [
                    wandb.Image(grid_image, caption="Iteration_" + str(epoch))]})

            epoch_loss = batch_loss / count_images if count_images > 0 else batch_loss
            epoch_loss_mse = batch_loss_mse / count_images if count_images > 0 else batch_loss_mse

            epoch_losses.append(epoch_loss)

            end_time = time()
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss, end_time - start_time, count_images))
            wandb.log({"Train/Loss_": epoch_loss, '_step_': epoch})
            wandb.log({"Train/Loss_MSE_": epoch_loss_mse, '_step_': epoch})

            # Save latest model
            torch.save({'model_weights': self.model.state_dict(), 'optimizer_weights': self.optimizer.state_dict()
                           , 'epoch': epoch}, os.path.join(self.client_path, 'latest_model.pt'))

            # Run validation
            if (epoch + 1) % self.val_interval == 0 and epoch > 0:
                self.test(self.model.state_dict(), self.val_ds, 'Val', self.optimizer.state_dict(), epoch)

        return self.best_weights, self.best_opt_weights

    def test(self, model_weights, test_data, task='Val', opt_weights=None, epoch=0):
        """
        :param model_weights: weights of the global model
        :return: dict
            metric_name : value
            e.g.:
             metrics = {
                'test_loss_rec': 0,
                'test_total': 0
            }
        """
        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()
        metrics = {
            task + '_loss_rec': 0,
            task + '_loss_mse': 0,
            task + '_loss_pl': 0,
        }
        test_total = 0

        with torch.no_grad():
            for data in test_data:
                x = data[0][0:1,:,:,:]
                x *=2
                x -=1 
                b, c, h, w = x.shape
                test_total += b
                x = x.to(self.device)
                # t, weights = self.scheduler.sample(x.shape[0], self.device)

                # self.eval_model = self.model().eval()
                print(f'X shape: {x.shape}')
                

                # sample_known, x_noisy_known, x_known = self.diffusion.p_sample_loop_known(self.test_model,
                                                                                            #  x.shape, x, clip_denoised=True, noise_level=500)
                sample_known, x_noisy_known, x_known = self.diffusion.sample_known(x,self.test_model, noise_level=300)

                sample_known_interpol, init_interpol, _, _ = self.diffusion.p_sample_loop_interpolation(self.test_model, x.shape, x, sample_known, 1-torch.abs(((x+1)/2)- ((sample_known+1)/2)), noise_level=50)
                #  = self.diffusion.training_losses(self.model, x, t)
                # x_, x_noisy, sample = self.diffusion.ddim_sample_loop(self.test_model, x.shape, noise=x)
                # import pdb 
                # pdb.set_trace()
                loss_rec = self.criterion_MSE(sample_known, x)
                loss_mse = self.criterion_MSE(sample_known, x)
                loss_pl = self.criterion_PL(sample_known, x)

                metrics[task + '_loss_rec'] += loss_rec.item() * x.size(0)
                metrics[task + '_loss_mse'] += loss_mse.item() * x.size(0)
                metrics[task + '_loss_pl'] += loss_pl.item() * x.size(0)

        img = x.detach().cpu()[0].numpy()
        # rec = sample.detach().cpu()[0].numpy()
        x_known_ = x_known.detach().cpu()[0].numpy()
        x_known_interpol = init_interpol.detach().cpu()[0].numpy()

        rec_known = sample_known.detach().cpu()[0].numpy()
        rec_known_interpl = sample_known_interpol.detach().cpu()[0].numpy()

        x_noisy_ = x_noisy_known.detach().cpu()[0].numpy()
        images = [img, x_known_, x_noisy_, rec_known]#, rec]
        # for im in images:
        #     print(im.min(), im.max())
        #     np.clip(im, 0, 1)
        #     im[0,0]=0
        #     im[0,1]=1

        elements = [img, x_noisy_, rec_known, np.abs((img+1)/2 - (rec_known+1)/2), x_known_interpol, rec_known_interpl, np.abs((img+1)/2-(rec_known_interpl+1)/2)]
        v_maxs = [1, 1, 1, 0.5, 1, 1, 0.5]
        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
        diffp.set_size_inches(len(elements) * 4, 4)
        for i in range(len(axarr)):
            axarr[i].axis('off')
            v_max = v_maxs[i]
            c_map = 'gray' if v_max == 1 else 'inferno'
            vmin = -1 if v_max == 1 else 0
            axarr[i].imshow(elements[i].transpose(1, 2, 0), vmin=vmin, vmax=v_max, cmap=c_map)

        wandb.log({task + '/Example_': [
                wandb.Image(diffp, caption="Iteration_" + str(epoch))]})
        # grid_image = np.hstack([x_known_, rec_known])

        # wandb.log({task + '/Example_': [
                # wandb.Image(grid_image, caption="Iteration_" + str(epoch))]})

        for metric_key in metrics.keys():
            metric_name = task + '/' + str(metric_key)
            metric_score = metrics[metric_key] / test_total
            wandb.log({metric_name: metric_score, '_step_': epoch})
        wandb.log({'lr': self.optimizer.param_groups[0]['lr'], '_step_': epoch})
        epoch_val_loss = metrics[task + '_loss_rec'] / test_total
        if task == 'Val':
            if epoch_val_loss < self.min_val_loss:
                self.min_val_loss = epoch_val_loss
                torch.save({'model_weights': model_weights, 'optimizer_weights': opt_weights, 'epoch': epoch},
                           os.path.join(self.client_path, 'best_model.pt'))
                self.best_weights = model_weights
                self.best_opt_weights = opt_weights
            self.early_stop = self.early_stopping(epoch_val_loss)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch_val_loss)