from core.Trainer import Trainer
from time import time
import os
import wandb
import logging
import matplotlib.pyplot as plt
from net_utils.simplex_noise import generate_noise, generate_simplex_noise
from optim.losses.image_losses import *
from torch.cuda.amp import GradScaler, autocast

"""
Implementation of the abstract class "Trainer" from core module.
Includes training and validation routines
"""
class PTrainer(Trainer):
    def __init__(self, training_params, model, data, device, log_wandb=True):
        super(PTrainer, self).__init__(training_params, model, data, device, log_wandb)
        self.val_interval = training_params['val_interval']

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
            self.model.load_state_dict(model_state, strict=False)  # load weights
            logging.info("[Trainer::weights]: ################ Model weights loaded ################")
        else: 
            logging.info("[Trainer::weights]: ################ No model weights were loaded ################")
        if opt_state is not None:
            try: 
                self.optimizer.load_state_dict(opt_state, strict=False)  # load optimizer
            except: 
                logging.warning("[Trainer::weights]: ################ Optimizer state could not be loaded ################")

        epoch_losses = []

        self.early_stop = False
        # to handle loss with mixed precision training
        scaler = GradScaler()

        for epoch in range(self.training_params['nr_epochs']):
            if start_epoch > epoch:
                continue
            if self.early_stop is True:
                logging.info("[Trainer::test]: ################ Finished training (early stopping) ################")
                break
            start_time = time()
            batch_loss, count_images = 1.0, 0

            for data in self.train_ds:
                # Input
                images = data[0].to(self.device)
                count_images += images.shape[0]
                transformed_images = self.transform(images) if self.transform is not None else images
                images = (images * 2) - 1
                transformed_images = (transformed_images * 2) - 1
                
                self.optimizer.zero_grad()
                
                # for mixed precision training
                with autocast(enabled=True):
                    # Create timesteps
                    timesteps = torch.randint(
                        0, self.model.train_scheduler.num_train_timesteps, (transformed_images.shape[0],), device=images.device
                    ).long()

                    # Generate random noise and noisy images
                    noise = generate_noise(self.model.train_scheduler.noise_type, images, self.model.train_scheduler.num_train_timesteps)
                    
                    # Get model prediction
                    pred = self.model(inputs=transformed_images, noise=noise, timesteps=timesteps)
                    transformed_images = (transformed_images + 1) / 2
                    images = (images + 1) / 2 

                    # cases for prediction_type = 'sample' or 'epsilon'
                    target = transformed_images if self.model.prediction_type == 'sample' else noise
                    pred = (pred + 1) / 2 if self.model.prediction_type == 'sample' else pred
                    loss = self.criterion_rec(pred.float(), target.float())

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                batch_loss += loss.item() * images.size(0)

            epoch_loss = batch_loss / count_images if count_images > 0 else batch_loss
            epoch_losses.append(epoch_loss)

            end_time = time()
            print('Epoch: {} \tTraining Loss: {:.6f} , computed in {} seconds for {} samples'.format(
                epoch, epoch_loss, end_time - start_time, count_images))
            wandb.log({"Train/Loss_": epoch_loss, '_step_': epoch})

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
                x = data[0]
                b, c, h, w = x.shape
                test_total += b
                x = x.to(self.device)
                x = (x * 2) - 1
                x_, _ = self.test_model.sample_from_image(x, noise_level=self.model.noise_level_recon)
                x = (x + 1) /2 
                x_ = (x_ + 1) / 2
                loss_rec = self.criterion_rec(x_, x)
                loss_mse = self.criterion_MSE(x_, x)
                loss_pl = self.criterion_PL(x_, x)

                metrics[task + '_loss_rec'] += loss_rec.item() * x.size(0)
                metrics[task + '_loss_mse'] += loss_mse.item() * x.size(0)
                metrics[task + '_loss_pl'] += loss_pl.item() * x.size(0)
                break

        rec = x_.detach().cpu()[0].numpy()
        img = x.detach().cpu()[0].numpy()

        elements = [img, rec, np.abs(img-rec)]
        v_maxs = [1, 1, 0.5]
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
