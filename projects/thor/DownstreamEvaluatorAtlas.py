import logging
import io
#
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.getLogger("matplotlib").setLevel(logging.WARNING)

import wandb
import plotly.graph_objects as go
import seaborn as sns
# import umap.umap_ as umap
#
from torch.nn import L1Loss
from torch.cuda.amp import autocast
from torchvision.transforms import transforms
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from torch.nn import MSELoss
from optim.losses import PerceptualLoss

#
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim as ssim2
from skimage import exposure
from scipy.ndimage.filters import gaussian_filter
from net_utils.simplex_noise import generate_noise, generate_simplex_noise

from PIL import Image
import cv2
import numpy as np
#
import lpips
#
from dl_utils import *
from optim.metrics import *
from optim.losses.image_losses import NCC
from core.DownstreamEvaluator import DownstreamEvaluator
import os
import copy
from model_zoo.vgg import VGGEncoder
from optim.losses.image_losses import CosineSimLoss



class PDownstreamEvaluator(DownstreamEvaluator):
    """
    Federated Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """

    def __init__(self, name, model, device, test_data_dict, checkpoint_path, global_=True):
        super(PDownstreamEvaluator, self).__init__(name, model, device, test_data_dict, checkpoint_path)

        self.criterion_rec = L1Loss().to(self.device)
        self.criterion_MSE = MSELoss().to(device)
        self.criterion_PL = PerceptualLoss(device=device)
        self.vgg_encoder = VGGEncoder().to(self.device)
        self.l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=True,
                                     lpips=True).to(self.device)

        # 71 - 570 - inf

        # self.l_pips_vgg = lpips.LPIPS(pretrained=True, net='vgg', use_dropout=False, eval_mode=False, spatial=False, lpips=True).to(self.device)
        # self.l_pips_alex = lpips.LPIPS(pretrained=True, net='alex', use_dropout=False, eval_mode=False, spatial=False, lpips=True).to(self.device)

        self.global_ = True

    def start_task(self, global_model):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
                   the model weights
        """
        # self.pseudo_healthy(global_model)

        self.pathology_localization(global_model, 3, 71, True)
        self.pathology_localization(global_model, 71, 570, True)
        self.pathology_localization(global_model, 570, 10000, True)
        
        # self.noise_images_plotting(global_model)
        # self.unconditional_sample(global_model)
        # self.image_reconstruction(global_model)
        
        # self.image_reconstruction(global_model)
        # self.anomaly_detection_intermediates_vis(global_model)
        
    def anomaly_detection_intermediates_vis(self, global_model):
        lpips_alex = lpips.LPIPS(net='alex')
        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()
        
        task_name = 'Pseudo_Healthy_Visualization'
        dataset_key = list(self.test_data_dict.keys())[0]
        image_caption_base = f"{dataset_key}_{str(self.model.inference_scheduler)[:4].lower()}"
        x_ = (next(iter(self.test_data_dict[dataset_key]))[0]).to(self.device)
        wandb.define_metric("noise_level")
        wandb.define_metric("*", step_metric="noise_level")

        with torch.no_grad():
            noise_levels = list(np.arange(0, self.model.train_scheduler.num_train_timesteps, 100))
            img_recs_list = []
            img_intermediates_list = []
            for noise_level_recon in noise_levels:
                anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly(copy.deepcopy(x_), noise_level=noise_level_recon)
                # x and x_rec comparison 
                x_rec = x_rec_dict['x_rec'] if 'x_rec' in x_rec_dict.keys() else torch.zeros_like(x)
                x_rec_ = torch.clamp(x_rec, 0, 1)
                diff_ = torch.abs(x_ - x_rec_)
                torch.from_numpy(anomaly_map)
                anomaly_map = torch.from_numpy(anomaly_map).to(self.device)
                imag_rec_tovis = torch.stack([x_] + [x_rec_] + [anomaly_map] +[diff_]).permute(1,0,2,3,4) # (B, S, C, H, W)
                x_labels = ['x', 'x_rec', 'anomaly_map', 'diff']
                y_labels = ["sample_" + str(i) for i in range(x_.shape[0])]
                self._batch_visualization(task_name, f"{image_caption_base}_noise_level={noise_level_recon}_pseudo_healthy_samples", x_labels, y_labels, imag_rec_tovis.detach().cpu(), col_cmaps=['gray', 'gray', 'plasma', 'plasma'], col_vmax=[None, None, 0.5, 0.5])
                img_recs_list.append(x_rec_)
                # metrics
                mae_list = []
                pl_list = []
                ssim_list = []
                for i in range(x_.shape[0]):
                    mae = torch.mean(torch.abs(x_rec_[i] - x_[i])).detach().cpu().numpy()
                    pl = np.squeeze(lpips_alex(x_rec_[i].cpu(), x_[i].cpu())).detach()
                    ssim_ = ssim(x_rec_[i].squeeze(0).cpu().numpy(), x_[i].squeeze(0).cpu().numpy(), data_range=1.)
                    mae_list.append(mae.item())
                    pl_list.append(pl.item())
                    ssim_list.append(ssim_.item())
                wandb.log(
                    {   
                        "noise_level": noise_level_recon,
                        f"{task_name}_metrics/mae": np.mean(mae_list),
                        f"{task_name}_metrics/pl": np.mean(pl_list),
                        f"{task_name}_metrics/ssim": np.mean(ssim_list)
                    }
                )
            img_recs_tovis = torch.stack([x_] + img_recs_list).permute(1,0,2,3,4) # (B, S, C, H, W)
            x_labels = ['x'] + [f'x_rec_{i}' for i in noise_levels]
            y_labels = ["sample_" + str(i) for i in range(x_.shape[0])]
            self._batch_visualization(task_name, f"{image_caption_base}_pseudo_healthy_samples", x_labels, y_labels, img_recs_tovis.detach().cpu())                          

                
    def image_reconstruction(self, global_model):
        lpips_alex = lpips.LPIPS(net='alex')
        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()
        
        dataset_key = list(self.test_data_dict.keys())[0]
        x_ = (next(iter(self.test_data_dict[dataset_key]))[0]).to(self.device)
        x = (x_ * 2) - 1
        task_name = 'Image_Reconstruction'
        image_caption_base = f"{dataset_key}_{str(self.model.inference_scheduler)[:4].lower()}"
        wandb.define_metric("noise_level")
        wandb.define_metric("*", step_metric="noise_level")

        with torch.no_grad():
            noise_levels = list(np.arange(0, self.model.train_scheduler.num_train_timesteps, 100))
            img_recs_list = []
            img_intermediates_list = []
            for noise_level_recon in noise_levels:
                x_rec, intermediates = self.model.sample_from_image(x, noise_level=noise_level_recon, save_intermediates=True, intermediate_steps=100)
                # x and x_rec comparison 
                x_rec_ = (x_rec + 1) / 2
                x_rec_ = torch.clamp(x_rec_, 0, 1)
                diff_ = torch.abs(x_ - x_rec_)
                imag_rec_tovis = torch.stack([x_] + [x_rec_] + [diff_]).permute(1,0,2,3,4) # (B, S, C, H, W)
                x_labels = ['x', 'x_rec', 'diff']
                y_labels = ["sample_" + str(i) for i in range(x.shape[0])]
                self._batch_visualization(task_name, f"{image_caption_base}_noise_level={noise_level_recon}_samples", x_labels, y_labels, imag_rec_tovis.detach().cpu(), col_cmaps=['gray', 'gray', 'plasma'], col_vmax=[None, None, 0.5])
                img_recs_list.append(x_rec_)
                # metrics
                mae_list = []
                pl_list = []
                ssim_list = []
                for i in range(x.shape[0]):
                    mae = torch.mean(torch.abs(x_rec_[i] - x_[i])).detach().cpu().numpy()
                    pl = np.squeeze(lpips_alex(x_rec_[i].cpu(), x_[i].cpu())).detach()
                    ssim_ = ssim(x_rec_[i].squeeze(0).cpu().numpy(), x_[i].squeeze(0).cpu().numpy(), data_range=1.)
                    mae_list.append(mae.item())
                    pl_list.append(pl.item())
                    ssim_list.append(ssim_.item())
                wandb.log(
                    {   
                        "noise_level": noise_level_recon,
                        f"{task_name}_metrics/mae": np.mean(mae_list),
                        f"{task_name}_metrics/pl": np.mean(pl_list),
                        f"{task_name}_metrics/ssim": np.mean(ssim_list)
                    }
                )
                # intermediates plotting
                intermediates_ = [torch.clamp((intermediate + 1) / 2, 0, 1) for intermediate in intermediates['z']]
                if len(intermediates_) < 10:
                    intermediates_ = [torch.zeros_like(intermediates_[0])] * (10 - len(intermediates_)) + intermediates_
                intermediates_ = torch.stack([x_] + intermediates_ + [x_rec_] + [diff_]) # (T=13, B, C, H, W)
                img_intermediates_list.append(intermediates_)
            img_intermediates_ = torch.stack(img_intermediates_list) # (N, T, B, C, H, W) 
            img_intermediates_ = img_intermediates_.permute(2,0,1,3,4,5) # (B, N, T, C, H, W)
            for i in range(x.shape[0]):
                img_intermediates_tovis = img_intermediates_[i] # (N, T, C, H, W)
                x_labels = ['x'] + [f'step_{noise_level}' for noise_level in reversed(noise_levels)] + ['x_rec', 'diff']
                y_labels = ["recon_from_" + str(noise_level) for noise_level in noise_levels]
                col_cmaps=['gray'] * (img_intermediates_tovis.shape[1] - 1) + ['plasma']
                col_vmax=[None] * (img_intermediates_tovis.shape[1] - 1) + [0.5]
                self._batch_visualization(task_name, f"{image_caption_base}_intermediates_samples_{i}", x_labels, y_labels, img_intermediates_tovis.detach().cpu(), col_cmaps=col_cmaps, col_vmax=col_vmax)
            img_recs_tovis = torch.stack([x_] + img_recs_list).permute(1,0,2,3,4) # (B, S, C, H, W)
            x_labels = ['x'] + [f'x_rec_{i}' for i in noise_levels]
            y_labels = ["sample_" + str(i) for i in range(x.shape[0])]
            self._batch_visualization(task_name, f"{image_caption_base}_reconstruction_samples", x_labels, y_labels, img_recs_tovis.detach().cpu())                          
            
            
    def unconditional_sample(self, global_model):
        task_name = 'Unconditional_Sampling'
        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()
        
        dataset_key = list(self.test_data_dict.keys())[0]
        x = (next(iter(self.test_data_dict[dataset_key]))[0]).to(self.device)
        noise = generate_noise(self.model.train_scheduler.noise_type, x, self.model.inference_scheduler.num_inference_steps)
        noise_unique = noise[0].unsqueeze(0).repeat(noise.shape[0], 1, 1, 1)
        self._sample_visualization(task_name, dataset_key, noise, noise_sampling="various_noise")
        self._sample_visualization(task_name, dataset_key, noise_unique, noise_sampling="same_noise")

    def _sample_visualization(self, task_name, dataset_key, noise, noise_sampling):
        image_caption_base = f"{dataset_key}_{noise_sampling}_{str(self.model.inference_scheduler)[:4].lower()}"

        with torch.no_grad():
            # samples plotting
            samples, intermediates = self.model.sample(input_noise = noise, noise_level=self.model.inference_scheduler.num_inference_steps-1, save_intermediates=True, intermediate_steps=100)
            samples_ = (samples + 1) / 2
            samples_ = torch.clamp(samples_, 0, 1)
            grid = make_grid(samples_, nrow=4, normalize=True, padding=2)[0:1,:,:]
            wandb.log({f"{task_name}/{image_caption_base}_samples": [wandb.Image(grid, caption=f"{image_caption_base}_samples")]})
            # intermediates plotting
            noise_ = (torch.clamp(noise, -3, 3) + 3)/6
            intermediates_ = [torch.clamp((intermediate + 1) / 2, 0, 1) for intermediate in intermediates]
            intermediates_ = torch.stack([noise_] + intermediates_ + [samples_]).permute(1,0,2,3,4) # (B, S, C, H, W)
            x_labels = ['noise'] + [f"step_{i}" for i in reversed(range(0, self.model.inference_scheduler.num_inference_steps, 100))] + ['sample']
            y_labels = ["sample_" + str(i) for i in range(intermediates_.shape[1])]
            self._batch_visualization(task_name, f"{image_caption_base}_intermediates", x_labels, y_labels, intermediates_.detach().cpu())
            
    def _batch_visualization(self, task_name, image_caption, x_labels, y_labels, batch_images, col_cmaps=None, col_vmax=None, padding=2):
        B, S, C, H, W = batch_images.shape
                
        fig, ax = plt.subplots(figsize=(S*1.2, B))
        total_W = S * W + (S - 1) * padding
        total_H = B * H + (B - 1) * padding
        ax.set_xlim(0, total_W)
        ax.set_ylim(total_H, 0)
        # default colormap and vmax for each column
        if col_cmaps is None:
            col_cmaps = ['gray'] * S
        if col_vmax is None:
            col_vmax = [None] * S
            
        # drawing each image block
        for i in range(B):
            for j in range(S):
                block = batch_images[i, j, :, :, :].reshape(H, W)
                
                x = j * (W + padding)
                y = i * (H + padding)

                ax.imshow(
                    block,
                    cmap=col_cmaps[j],
                    vmin=0,
                    vmax=col_vmax[j],
                    extent=[x, x+W, y+H, y]
                )
        ax.axis('off')

        # column labels
        for j in range(S):
            x = j * (W + padding) + W // 2
            ax.text(
                x, -5,
                x_labels[j],
                ha='center',
                va='bottom',
                fontsize=8
            )

        # row labels
        for i in range(B):
            y = i * (H + padding) + H // 2
            ax.text(
                -5, y,
                y_labels[i],
                ha='right',
                va='center',
                fontsize=8
            )

        plt.tight_layout()
        plt.show()
        
        with io.BytesIO() as buf:
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            img = Image.open(buf)
            wandb.log({f"{task_name}/{image_caption}": [wandb.Image(img, caption=image_caption)]}) 
            
    def noise_images_plotting(self, global_model):
        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()
        
        dataset_key = list(self.test_data_dict.keys())[0]
        x_ = (next(iter(self.test_data_dict[dataset_key]))[0]).to(self.device)
        x = (x_ * 2) - 1
        task_name = 'noisy_images'
        image_caption_base = f"{dataset_key}_{str(self.model.inference_scheduler)[:4].lower()}"

        with torch.no_grad():
            noise_levels = list(np.arange(0, self.model.train_scheduler.num_train_timesteps, 100))
            img_noisy_list = []
            for noise_level_recon in noise_levels:
                noise = generate_noise(self.model.train_scheduler.noise_type, x_, noise_level_recon)
                img_noisy = self.model.train_scheduler.add_noise(original_samples=x, noise=noise, timesteps=torch.tensor([noise_level_recon], device=self.device))
                img_noisy = self._normalize_per_image(img_noisy)
                img_noisy_list.append(img_noisy)
            img_noisy_ = torch.stack([x_] + img_noisy_list) # (S, B, C, H, W)
            img_noisy_tovis = img_noisy_.permute(1,0,2,3,4)  # (B, S, C, H, W)
            x_labels = ['x'] + [f'step_{i}' for i in noise_levels]
            y_labels = ["sample_" + str(i) for i in range(x.shape[0])]
            self._batch_visualization(task_name, f"{image_caption_base}_{self.model.train_scheduler.noise_type}_noisy_images", x_labels, y_labels, img_noisy_tovis.detach().cpu())                          
 

    def _normalize_per_image(self,x):
        # x: [B, 1, H, W]
        B = x.shape[0]
        x_flat = x.view(B, -1)

        min_val = x_flat.min(dim=1)[0].view(B,1,1,1)
        max_val = x_flat.max(dim=1)[0].view(B,1,1,1)

        x_norm = (x - min_val) / (max_val - min_val + 1e-8)
        return x_norm

    def _log_visualization(self, to_visualize, i, count, task_name, dataset_key):
        """
        Helper function to log images and masks to wandb
        :param: to_visualize: list of dicts of images and their configs to be visualized
            dict needs to include:
            - tensor: image tensor
            dict may include:
            - title: title of image
            - cmap: matplotlib colormap name
            - vmin: minimum value for colorbar
            - vmax: maximum value for colorbar
        :param: epoch: current epoch
        """
        diffp, axarr = plt.subplots(1, len(to_visualize), gridspec_kw={'wspace': 0, 'hspace': 0},
                                    figsize=(len(to_visualize) * 4, 4))
        for idx, dict in enumerate(to_visualize):
            if 'title' in dict:
                axarr[idx].set_title(dict['title'])
            axarr[idx].axis('off')
            tensor = dict['tensor'][i].cpu().detach().numpy().squeeze() if isinstance(dict['tensor'], torch.Tensor) else \
            dict['tensor'][i].squeeze()
            axarr[idx].imshow(tensor, cmap=dict.get('cmap', 'gray'), vmin=dict.get('vmin', 0), vmax=dict.get('vmax', 1))
        diffp.set_size_inches(len(to_visualize) * 4, 4)

        wandb.log({f'{task_name}/Example_{dataset_key}_{count}': [wandb.Image(diffp, caption=f"{dataset_key}_{count}")]})


    def find_mask_size_thresholds(self, dataset):
        """
        :param dataset: dataset to find mask size thresholds
        :return: lower and upper tail thresholds
        """
        mask_sizes = []
        for _, data in enumerate(dataset):
            if 'dict' in str(type(data)) and 'images' in data.keys():
                data0 = data['images']
            else:
                data0 = data[0]
            x = data0.to(self.device)
            masks = data[1].to(self.device)
            masks[masks > 0] = 1

            for i in range(len(x)):
                if torch.sum(masks[i][0]) > 1:
                    mask_sizes.append(torch.sum(masks[i][0]).item())

        unique_mask_sizes = np.unique(mask_sizes)
        print(type(unique_mask_sizes))
        lower_tail_threshold = np.percentile(unique_mask_sizes, 25)
        upper_tail_threshold = np.percentile(unique_mask_sizes, 75)

        _ = plt.figure()
        # plt.figure()
        plt.hist(mask_sizes, bins=100)
        plt.xlabel('Mask Sizes')
        plt.ylabel('Frequency')
        plt.title('Histogram of Mask Sizes')

        plt.axvline(lower_tail_threshold, color='r', linestyle='--', label=f'25th Percentile: {lower_tail_threshold}')
        plt.axvline(upper_tail_threshold, color='g', linestyle='--', label=f'75th Percentile: {upper_tail_threshold}')
        print(lower_tail_threshold, upper_tail_threshold)
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        wandb.log({"Anomaly/Mask sizes1": [wandb.Image(Image.open(buf), caption="Mask Sizes")]})

        plt.clf()


    def pseudo_healthy(self, global_model):
        """
        Validation of downstream tasks
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        logging.info("################ Pseudo Healthy TEST #################")
        # lpips_alex = lpips.LPIPS(net='vgg')  # best forward scores
        # lpips_alex = lpips.LPIPS(pretrained=True, net='alex', use_dropout=True, eval_mode=True, spatial=True, lpips=True).to(self.device)
        lpips_alex = lpips.LPIPS(net='alex').to(self.device)


        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        self.model.load_state_dict(global_model)
        self.model.eval()
        metrics = {
            'MAE': [],
            'LPIPS': [],
            'SSIM': [],
        }
        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'MAE': [],
                'LPIPS': [],
                'SSIM': [],
            }
            pred = []
            gt = []
            logging.info('DATASET: {}'.format(dataset_key))
            for idx, data in enumerate(dataset):
                                # New per batch
                if 'dict' in str(type(data)) and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                
                anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly(copy.deepcopy(x))
                x_rec = x_rec_dict['x_rec'] if 'x_rec' in x_rec_dict.keys() else torch.zeros_like(x)
                x_rec = torch.clamp(x_rec, 0, 1)

                to_visualize = [
                    {'title': 'x', 'tensor': x},
                    {'title': 'x_rec', 'tensor': x_rec},
                    {'title': f'Anomaly  map {anomaly_map.max():.3f}', 'tensor': anomaly_map, 'cmap': 'plasma',
                     'vmax': 0.999},
                ]

                for i in range(len(x)):
                        count = str(idx * len(x) + i)
                        # Example visualizations
                        if int(count) % 10 == 0 or int(count) in [0, 66, 325, 352, 545, 548, 231, 609, 616, 11, 254,
                                                                  539, 165, 545, 550, 92, 616, 628, 630, 636, 651]:
                            self._log_visualization(to_visualize, i, count, task_name='Anomaly_masks', dataset_key=dataset_key)

                        x_i = x[i][0]
                        rec_2_i = x_rec[i][0]

                        # Similarity metrics: x_rec vs. x
                        loss_mae = torch.mean(torch.abs(rec_2_i - x_i))
                        test_metrics['MAE'].append(loss_mae.cpu().detach().numpy())
                        loss_lpips = lpips_alex(x_i, rec_2_i)
                        test_metrics['LPIPS'].append(loss_lpips.cpu().detach().numpy())
                        ssim_ = ssim(rec_2_i.cpu().detach().numpy(), x_i.cpu().detach().numpy(), data_range=1.)
                        test_metrics['SSIM'].append(ssim_)

            for metric in test_metrics:
                logging.info('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                                         np.nanstd(test_metrics[metric])))
                metrics[metric].append(test_metrics[metric])

        logging.info('Writing plots...')

        for metric in metrics:
            fig_bp = go.Figure()
            x = []
            y = []
            for idx, dataset_values in enumerate(metrics[metric]):
                dataset_name = list(self.test_data_dict)[idx]
                for dataset_val in dataset_values:
                    y.append(dataset_val)
                    x.append(dataset_name)

            fig_bp.add_trace(go.Box(
                y=y,
                x=x,
                name=metric,
                boxmean='sd'
            ))
            title = 'score'
            fig_bp.update_layout(
                yaxis_title=title,
                boxmode='group',  # group together boxes of the different traces for each value of x
                yaxis=dict(range=[0, 1]),
            )
            fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)

            wandb.log({"Reconstruction_Metrics(Healthy)_" + self.name + '_' + str(metric): fig_bp})

   
    def pathology_localization(self, global_model, threshold_low, threshold_high, perc_flag=False):
        """
        Validation of downstream tasks
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        logging.info(f"################ Stroke Anomaly Detection {threshold_low} - {threshold_high} #################")
        lpips_alex = lpips.LPIPS(net='alex')

        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()
        metrics = {
            'MAE': [],
            'LPIPS': [],
            'SSIM': [],
        }
        pred_dict = dict()

        for dataset_key in self.test_data_dict.keys():
            pred_ = []
            label_ = []
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'MAE': [],
                'LPIPS': [],
                'SSIM': [],
            }
            global_counter = 0
            threshold_masks = []
            anomalous_pred = []
            healthy_pred = []

            logging.info('DATASET: {}'.format(dataset_key))

            for idx, data in enumerate(dataset):

                if idx not in [3,8,15,17,18,22,81,101,381,440,530,597,598,602,636, 66, 550, 616, 548, 545, 325]:
                    continue

                # Call this to get the mask size thresholds for the dataset
                # self.find_mask_size_thresholds(dataset)

                # New per batch
                if 'dict' in str(type(data)) and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                masks = data[1].to(self.device)
                masks[masks > 0] = 1
                
                if torch.sum(masks[0][0]) <= threshold_low or torch.sum(masks[0][0]) > threshold_high: 
                    continue # get the desired sizes of anomalies
                anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly(copy.deepcopy(x))
                x_rec = x_rec_dict['x_rec'] if 'x_rec' in x_rec_dict.keys() else torch.zeros_like(x)
                x_rec = torch.clamp(x_rec, 0, 1)

                # TODO: add one anomaly map of difference between input and x_rec
                to_visualize = [
                    {'title': 'x', 'tensor': x},
                    {'title': 'x_rec', 'tensor': x_rec},
                    {'title': f'Anomaly_map {anomaly_map.max():.3f}(max)', 'tensor': anomaly_map, 'cmap': 'plasma',
                     'vmax': anomaly_map.max()},
                    {'title': 'gt', 'tensor': masks, 'cmap': 'plasma'}
                ]
                
                if 'mask' in x_rec_dict.keys():
                    x_ = x.cpu().detach().numpy()
                    masked_input = x_rec_dict['mask'] + x_
                    masked_input[masked_input>1]=1

                    # to_visualize.append({'title': 'Rec Orig', 'tensor': x_rec_dict['x_rec_orig'], 'cmap': 'gray'})
                    # to_visualize.append({'title': 'Res Orig', 'tensor': x_rec_dict['x_res'], 'cmap': 'plasma',
                    #                     'vmax': x_rec_dict['x_res'].max()})
                    to_visualize.append({'title': 'Masked_input', 'tensor': masked_input, 'cmap': 'plasma', 'vmax': 0.5})

                for i in range(len(x)):
                    if torch.sum(masks[i][0]) > threshold_low and torch.sum(
                            masks[i][0]) <= threshold_high:  # get the desired sizes of anomalies
                        count = str(idx * len(x) + i)
                        # Don't use images with large black artifacts:
                        if int(count) in [100, 105, 112, 121, 186, 189, 210, 214, 345, 382, 424, 425, 435, 434, 441,
                                          462, 464, 472, 478, 504]:
                            print("skipping ", count)
                            continue

                        # Example visualizations
                        if int(count) % 1000 == 0 or int(count) in [3,8,15,17,18,22,81,101,381,440,530,597,598,602,636, 66, 550, 616, 548, 545, 325]: #[0, 66, 325, 352, 545, 548, 231, 609, 616, 11, 254,
                                                                #   539, 165, 545, 550, 92, 616, 628, 630, 636, 651]:
                            self._log_visualization(to_visualize, i, count, task_name='Anomaly_masks', dataset_key=dataset_key)

                        x_i = x[i][0]
                        rec_2_i = x_rec[i][0]

                        res_2_i_np = anomaly_map[i][0]
                        anomalous_pred.append(anomaly_score[i][0])

                        pred_.append(res_2_i_np)
                        label_.append(masks[i][0].cpu().detach().numpy())

                        # Similarity metrics: x_rec vs. x
                        loss_mae = torch.mean(torch.abs(rec_2_i - x_i))
                        test_metrics['MAE'].append(loss_mae.item())
                        loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), rec_2_i.cpu()).detach().numpy())
                        test_metrics['LPIPS'].append(loss_lpips)
                        ssim_ = ssim(rec_2_i.cpu().detach().numpy(), x_i.cpu().detach().numpy(), data_range=1.)
                        test_metrics['SSIM'].append(ssim_)

                    elif torch.sum(
                            masks[i][0]) <= 1:  # use slices without anomalies as "healthy" examples on same domain
                        res_2_i_np_healthy = anomaly_map[i][0]# * combined_mask[i][0].cpu().detach().numpy()
                        healthy_pred.append(anomaly_score[i][0])
                # if len(pred_) > 5:
                    # print(f'Done with the validaiton for now... {len(pred_)}')
                    # break
                # break

            pred_dict[dataset_key] = (pred_, label_)

            for metric in test_metrics:
                logging.info('{}: {} mean: {} +/- {}'.format(dataset_key, metric, np.nanmean(test_metrics[metric]),
                                                             np.nanstd(test_metrics[metric])))
                metrics[metric].append(test_metrics[metric])

        for dataset_key in self.test_data_dict.keys():
            # Get some stats on prediction set
            pred_ood, label_ood = pred_dict[dataset_key]
            predictions = np.asarray(pred_ood)
            labels = np.asarray(label_ood)
            predictions_all = np.reshape(np.asarray(predictions), (len(predictions), -1))  # .flatten()
            labels_all = np.reshape(np.asarray(labels), (len(labels), -1))  # .flatten()
            print(f'Nr of preditions: {predictions_all.shape}')
            print(
                f'Predictions go from {np.min(predictions_all)} to {np.max(predictions_all)} with mean: {np.mean(predictions_all)}')
            print(f'Labels go from {np.min(labels_all)} to {np.max(labels_all)} with mean: {np.mean(labels_all)}')
            print('Shapes {} {} '.format(labels.shape, predictions.shape))

            # Compute global anomaly localization metrics
            dice_scores = []

            auprc_, _, _, _ = compute_auprc(predictions_all, labels_all)
            logging.info(f'Global AUPRC score: {auprc_}')
            wandb.log({f'Metrics/{threshold_low}_Global_AUPRC_{dataset_key}': auprc_})

            # Compute dice score for linear thresholds from 0 to 1
            ths = np.linspace(0, 1, 101)
            for dice_threshold in ths:
                dice = compute_dice(copy.deepcopy(predictions_all), copy.deepcopy(labels_all), dice_threshold)
                dice_scores.append(dice)
            highest_score_index = np.argmax(dice_scores)
            highest_score = dice_scores[highest_score_index]

            logging.info(f'Global highest DICE: {highest_score}')
            wandb.log({f'Metrics/{threshold_low}_Global_highest_DICE': highest_score})

        # Plot box plots over the metrics per image
        logging.info('Writing plots...')
        for metric in metrics:
            fig_bp = go.Figure()
            x = []
            y = []
            for idx, dataset_values in enumerate(metrics[metric]):
                dataset_name = list(self.test_data_dict)[idx]
                for dataset_val in dataset_values:
                    y.append(dataset_val)
                    x.append(dataset_name)

            fig_bp.add_trace(go.Box(
                y=y,
                x=x,
                name=metric,
                boxmean='sd'
            ))
            title = 'score'
            fig_bp.update_layout(
                yaxis_title=title,
                boxmode='group',  # group together boxes of the different traces for each value of x
                yaxis=dict(range=[0, 1]),
            )
            fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)
            wandb.log({f'Metrics/{threshold_low}_{self.name}_{metric}': fig_bp})