# Based on Inferer module from MONAI:
# -----------------------------------------------------------------------------------------------
# Implements two different methods:
#   1). AnoDDPM: Wyatt et.: Anoddpm: "Anomaly detection with denoising diffusion probabilistic models using simplex
# noise." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops 650–656535, (2022).
#   2) AutoDDPM: CI Bercea et. al.: "Mask, stitch, and re-sample: Enhancing robustness and generalizability in
#   anomaly detection through automatic diffusion models." ICML Workshops, (2023).
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from net_utils.simplex_noise import generate_noise
from net_utils.nets.diffusion_unet import DiffusionModelUNet
from net_utils.schedulers.ddpm import DDPMScheduler
from net_utils.schedulers.ddim import DDIMScheduler

import wandb
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib").setLevel(logging.WARNING)
import copy

from tqdm import tqdm
has_tqdm = True
from skimage import exposure
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import median_filter, percentile_filter, grey_dilation, grey_closing, maximum_filter, grey_opening

from skimage.exposure import match_histograms
from skimage.morphology import square
from skimage.morphology import dilation, closing, area_closing, area_opening
from skimage.segmentation import flood_fill as segment_
from scipy import ndimage
from scipy import stats

import lpips
import cv2
from torch.cuda.amp import autocast


class DDPM(nn.Module):

    def __init__(self, spatial_dims=2,
                 in_channels=1,
                 out_channels=1,
                 num_channels=(128, 256, 256),
                 attention_levels=(False, True, True),
                 num_res_blocks=1,
                 num_head_channels=256,
                 train_scheduler="ddpm",
                 inference_scheduler="ddpm",
                 inference_steps=1000,
                 noise_level_recon=300,
                 noise_type="gaussian",
                 prediction_type="epsilon",
                 threshold_low=1,
                 threshold_high=10000,
                 inference_type='ano',
                 t_harmonization=[700, 600, 500, 400, 300, 150, 50], # Gausian, for Simplex use every 50 epochs 
                 t_visualization=[700,650,600,550,500,450,400,350,300,250,200,150,100,50,0],
                 image_path="",):
        super().__init__()
        self.unet = DiffusionModelUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_channels=num_channels,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            num_head_channels=num_head_channels,
        )
        self.inference_type = inference_type
        self.t_harmonization = t_harmonization
        self.t_visualization = t_visualization
        self.noise_level_recon = noise_level_recon
        self.prediction_type = prediction_type
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.image_path = image_path
        self.img_ct = 0

        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        # LPIPS for perceptual anomaly maps
        self.l_pips_sq = lpips.LPIPS(pretrained=True, pnet_rand=False, net='squeeze', eval_mode=True, spatial=True, lpips=True).to(self.device)

        # set up scheduler and timesteps
        if train_scheduler == "ddim":
            print('****** DIFFUSION: Using DDIM Scheduler ******')
            self.train_scheduler = DDIMScheduler(
                num_train_timesteps=1000, noise_type=noise_type, prediction_type=prediction_type)
        elif train_scheduler == 'ddpm':
            print('****** DIFFUSION: Using DDPM Scheduler ******')
            self.train_scheduler = DDPMScheduler(
                num_train_timesteps=1000, noise_type=noise_type, prediction_type=prediction_type)
        else:
            raise NotImplementedError(f"{train_scheduler} does is not implemented for {self.__class__}")

        if inference_scheduler == "ddim":
            print('****** DIFFUSION: Using DDIM Scheduler ******')
            self.inference_scheduler = DDIMScheduler(
                num_train_timesteps=1000, noise_type=noise_type, prediction_type=prediction_type)
        else:
            print('****** DIFFUSION: Using DDPM Scheduler ******')
            self.inference_scheduler = DDPMScheduler(
                num_train_timesteps=1000, noise_type=noise_type, prediction_type=prediction_type)

        self.inference_scheduler.set_timesteps(inference_steps)

    def forward(self, inputs, noise=None, timesteps=None, condition=None):
        # only for torch_summary to work
        if noise is None:
            noise = torch.randn_like(inputs)
        if timesteps is None:
            timesteps = torch.randint(0, self.train_scheduler.num_train_timesteps,
                                      (inputs.shape[0],), device=inputs.device).long()

        noisy_image = self.train_scheduler.add_noise(
            original_samples=inputs, noise=noise, timesteps=timesteps)
        return self.unet(x=noisy_image, timesteps=timesteps, context=condition)
    
    
    def get_anomaly_mask(self, x, x_rec, hist_eq=False, retPerLayer=False):
        x_res = self.compute_residual(x, x_rec, hist_eq=hist_eq)
        lpips_mask = self.get_saliency(x, x_rec, retPerLayer=retPerLayer).clip(0,1)

        x_res2 = np.asarray([(x_res[i] / (np.percentile(x_res[i], 95) + 1e-8)) for i in range(x_res.shape[0])]).clip(0, 1)

        combined_mask_np = lpips_mask * x_res #+ x_res) / 2
        combined_mask_np2 = (lpips_mask * x_res) # x_res2
        # # anomalous: high value, healthy: low value
        # combined_mask_np = area_opening((combined_mask_np * 255).astype(np.uint8)) / 255.0#, square(7))
        # combined_mask_np = closing((combined_mask_np * 255).astype(np.uint8), footprint=np.ones(9,9)) / 255.0#, square(7))
        # # combined_mask_np = ndimage.grey_dilation((combined_mask_np * 255).astype(np.uint8), size=(3)) / 255.0#, square(7))
        combined_mask = torch.Tensor(combined_mask_np).to(self.device)
        # combined_mask = self.dilate_masks(combined_mask)
        combined_mask2 = torch.Tensor(combined_mask_np2).to(self.device)

        combined_mask = (combined_mask / (torch.max(combined_mask) + 1e-8)).clip(0,1) 
        # x_res_neg = (x-x_rec)
        return combined_mask, combined_mask2, torch.Tensor(x_res).to(self.device)
        # return torch.Tensor(x_res).to(self.device), torch.Tensor(x_res).to(self.device), torch.Tensor(x_res).to(self.device)

    
    def get_region_anomaly_mask(self, ano_map, kernel_size=13):
        # input_image_ = (np.squeeze(copy.deepcopy(input_image).cpu().detach().numpy())*255).astype(np.uint8)
        final_anomaly_map = (grey_closing(ano_map, size=(1,1,kernel_size,kernel_size), mode='nearest'))#+ ano_map)/2
        final_anomaly_map = (grey_dilation(final_anomaly_map, size=(1,1,kernel_size,kernel_size), mode='nearest') + ano_map)/2
        final_anomaly_map = final_anomaly_map.clip(0,1)
        # final_anomaly_map = ((2**final_anomaly_map)-1).clip(0,1)
        return final_anomaly_map

    def print_intermediates(self, intermediates, title='Intermediates', img_ct=0, vmax=1):
        elements = [inter.cpu().detach().numpy() for inter in intermediates]
        v_maxs = [vmax for inter in intermediates]
        if len(elements) > 2:
            diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
            diffp.set_size_inches(len(elements) * 4, 4)
            for i in range(len(axarr)):
                axarr[i].axis('off')
                v_max = v_maxs[i]
                c_map = 'gray' if v_max == 1 else 'plasma'
                axarr[i].imshow(np.squeeze(elements[i]), vmin=0, vmax=v_max, cmap=c_map)

            wandb.log({'Diffusion' + '/' + title: [
                    wandb.Image(diffp, caption="Iteration_" + str(img_ct))]})
            

    def get_thor_anomaly(self, inputs):
        x_rec, z_dict = self.sample_from_image_interpol(inputs, noise_level=self.noise_level_recon, save_intermediates=True, intermediate_steps=self.intermediate_steps, t_harmonization=self.t_harmonization, t_visualization=self.t_visualization)
        # self.print_intermediates(z_dict['inter_ddpm'], 'Intermediates (DDPM)', self.img_ct, 1)
        # self.print_intermediates(z_dict['z'], 'Intermediates (THOR)', self.img_ct, 1)
        # self.print_intermediates(z_dict['inter_gt'], 'Intermediates (GT)', self.img_ct, 1)
        # self.print_intermediates(z_dict['inter_res'], 'Intermediates (Res_THOR)', self.img_ct, 0.999)
        # self.print_intermediates(z_dict['inter_res_ddpm'], 'Intermediates (Res_DDPM)', self.img_ct, 0.999)
        # self.print_intermediates(z_dict['inter_res_mix'], 'Intermediates (Res_MIX)', self.img_ct, 0.999)

        x_rec = torch.clamp(x_rec, 0, 1)


        np_res = [inter.cpu().detach().numpy() for inter in z_dict['inter_res']]
        x_rec_refined = x_rec 
        self.img_ct += 1 
        # anomaly_maps = self.get_anomaly_mask(inputs, x_rec_refined, hist_eq=False)[loss_idx].cpu().detach().numpy()
        x_res =  np_res[-1].clip(0, 0.999)
        anomaly_maps = x_res 
        masked_input = x_res
        anomaly_scores = np.mean(anomaly_maps, axis=(1, 2, 3), keepdims=True)

        return anomaly_maps, anomaly_scores, {'x_rec': x_rec_refined, 'mask': masked_input, 'x_res': x_res,
                                            'x_rec_orig': x_rec}
    

    def get_anomaly(self, inputs, noise_level=250):
        if self.inference_type == 'thor':
            return self.get_thor_anomaly(inputs)
        x_rec, _ = self.sample_from_image(inputs, self.noise_level_recon)
        x_rec_ = x_rec.cpu().detach().numpy()
        x_ = inputs.cpu().detach().numpy()
        anomaly_maps = np.abs(x_ - x_rec_)
        anomaly_scores = np.mean(anomaly_maps, axis=(1, 2, 3), keepdims=True)
        # anomaly_maps, anomaly_score = self.compute_anomaly(inputs, x_rec)
        return anomaly_maps, anomaly_scores, {'x_rec': x_rec}

    def compute_anomaly(self, x, x_rec):
        anomaly_maps = []
        for i in range(len(x)):
            x_res, saliency = self.compute_residual(x[i][0], x_rec[i][0])
            anomaly_maps.append(x_res*saliency)
        anomaly_maps = np.asarray(anomaly_maps)
        anomaly_scores = np.mean(anomaly_maps, axis=(1, 2, 3), keepdims=True)
        return anomaly_maps, anomaly_scores

    def compute_residual(self, x, x_rec, hist_eq=False):
        """
        :param x_rec: reconstructed image
        :param x: original image
        :param hist_eq: whether to perform histogram equalization
        :return: residual image
        """
        if hist_eq:
            x_rescale = exposure.equalize_adapthist(x.cpu().detach().numpy())
            x_rec_rescale = exposure.equalize_adapthist(x_rec.cpu().detach().numpy())
            x_res = np.abs(x_rec_rescale - x_rescale)
        else:
            x_res = np.abs(x_rec.cpu().detach().numpy() - x.cpu().detach().numpy())

        return x_res

    def lpips_loss(self, anomaly_img, ph_img, retPerLayer=False):
        """
        :param anomaly_img: anomaly image
        :param ph_img: pseudo-healthy image
        :param retPerLayer: whether to return the loss per layer
        :return: LPIPS loss
        """
        if len(ph_img.shape) == 2:
            ph_img = torch.unsqueeze(torch.unsqueeze(ph_img, 0), 0)
            anomaly_img = torch.unsqueeze(torch.unsqueeze(anomaly_img, 0), 0)
        

        anomaly_img = ((anomaly_img * 2) - 1).repeat(1,3,1,1)
        ph_img = ((ph_img * 2) - 1).repeat(1,3,1,1)

        loss_lpips = self.l_pips_sq(anomaly_img, ph_img, normalize=True, retPerLayer=retPerLayer)
        if retPerLayer:
            loss_lpips = loss_lpips[1][0]
        return loss_lpips.cpu().detach().numpy()

    def get_saliency(self, x, x_rec, retPerLayer=False):
        saliency = self.lpips_loss(x, x_rec, retPerLayer)
        saliency = gaussian_filter(saliency, sigma=2)
        return saliency

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        noise_level: int | None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        verbose: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            noise_level: noising step until which noise is added before sampling
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            verbose: if true, prints the progression bar of the sampling process.
        """
        image = input_noise
        timesteps = self.inference_scheduler.get_timesteps(noise_level)
        if verbose and has_tqdm:
            progress_bar = tqdm(timesteps)
        else:
            progress_bar = iter(timesteps)
        intermediates = []
        for t in progress_bar:
            # 1. predict noise model_output
            model_output = self.unet(
                image, timesteps=torch.Tensor((t,)).to(input_noise.device), context=conditioning
            )

            # 2. compute previous image: x_t -> x_t-1
            image, orig_image = self.inference_scheduler.step(model_output, t, image)
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(orig_image)
        if save_intermediates:
            return image, intermediates
        else:
            return image, None

    @torch.no_grad()
    # function to noise and then sample from the noise given an image to get healthy reconstructions of anomalous input images
    def sample_from_image(
        self,
        inputs: torch.Tensor,
        noise_level: int | None = 500,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        verbose: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Sample to specified noise level and use this as noisy input to sample back.
        Args:
            inputs: input images, NxCxHxW[xD]
            noise_level: noising step until which noise is added before 
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            verbose: if true, prints the progression bar of the sampling process.
        """
        noise = generate_noise(
            self.train_scheduler.noise_type, inputs, noise_level)

        t = torch.full((inputs.shape[0],),
                       noise_level, device=inputs.device).long()
        noised_image = self.train_scheduler.add_noise(
            original_samples=inputs, noise=noise, timesteps=t)
        image, intermediates = self.sample(input_noise=noised_image, noise_level=noise_level, save_intermediates=save_intermediates,
                            intermediate_steps=intermediate_steps, conditioning=conditioning, verbose=verbose)
        return image, {'z': intermediates}
    
    @torch.no_grad()
    # function to noise and then sample from the noise given an image to get healthy reconstructions of anomalous input images
    def sample_from_image_interpol(
        self,
        inputs: torch.Tensor,
        noise_level: int | None = 500,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        t_harmonization: [int] | None = [700, 600, 500, 400, 300, 150, 50],
        t_visualization: [int] | None = [700,650,600,550,500,450,400,350,300,250,200,150,100,50,0],
        conditioning: torch.Tensor | None = None,
        verbose: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Sample to specified noise level and use this as noisy input to sample back.
        Args:
            inputs: input images, NxCxHxW[xD]
            noise_level: noising step until which noise is added before 
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            verbose: if true, prints the progression bar of the sampling process.
        """

        loss_idx = 0
        do_hmatching = False


        noise = generate_noise(
            self.train_scheduler.noise_type, inputs, noise_level)

        t = torch.full((inputs.shape[0],),
                       noise_level, device=inputs.device).long()
        input_noise = self.train_scheduler.add_noise(
            original_samples=inputs, noise=noise, timesteps=t)
        image = input_noise
        input_noise_ddpm = self.train_scheduler.add_noise(
            original_samples=inputs, noise=noise, timesteps=t)
        image_ddpm = input_noise_ddpm

        timesteps = self.inference_scheduler.get_timesteps(noise_level)
        if verbose and has_tqdm:
            progress_bar = tqdm(timesteps)
        else:
            progress_bar = iter(timesteps)
        intermediates = []
        intermediates_ddpm = []
        intermediates_gt = []
        intermediates_res = []
        intermediates_res_ddpm = []
        intermediates_res_mix = []
        for t in progress_bar:

            ## DDPM PATH
            # 1. predict noise model_output
            # model_output_ddpm = self.unet(image_ddpm, timesteps=torch.Tensor((t,)).to(input_noise.device), context=conditioning)
            # 2. compute previous image: x_t -> x_t-1
            # image_ddpm, orig_image_ddpm = self.inference_scheduler.step(model_output_ddpm, t, image_ddpm)

            ## THOR PATH
            # 1. predict noise model_output
            model_output = self.unet(image, timesteps=torch.Tensor((t,)).to(input_noise.device), context=conditioning)
            # 2. compute previous image: x_t -> x_t-1
            image, orig_image = self.inference_scheduler.step(model_output, t, image)

            ## PATHOLOGY PATH (Reverse)
            # noise_gt = self.train_scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=t)
            # model_output_gt = self.unet(noise_gt, timesteps=torch.Tensor((t,)).to(input_noise.device), context=conditioning)
            # image_gt, orig_image_gt = self.inference_scheduler.step(model_output_gt, t, noise_gt)

            if save_intermediates and (t in t_harmonization or t in t_visualization):

                # intermediates_ddpm.append(orig_image_ddpm)
                intermediates.append(orig_image)
                # intermediates_gt.append(orig_image_gt)

                res_thor = self.get_anomaly_mask(copy.deepcopy(orig_image), copy.deepcopy(inputs), hist_eq=do_hmatching)[loss_idx]  # orig_image, orig_image_gt
                # res_ddpm = self.get_anomaly_mask(copy.deepcopy(orig_image), copy.deepcopy(inputs), hist_eq=do_hmatching)[2]  # orig_image, orig_image_gt

                # res_ddpm = res_ddpm.cpu().detach().numpy()
                res = res_thor
                
                # resnp = stats.hmean(np.stack(intermediates_res), axis=0) if len(intermediates_res) > 1 else copy.deepcopy(res).cpu().detach().numpy()
                resnp = res.cpu().detach().numpy()

                res_mix = resnp

                region_anomaly_map = self.get_region_anomaly_mask(res_mix) # Dilate the mask to exclude pathological areas 
                res = torch.Tensor(region_anomaly_map).to(self.device)
                res =  ((res)).clip(0,1)
                intermediates_res.append(res_mix)
                # intermediates_res_ddpm.append(res_ddpm)
                intermediates_res_mix.append(((region_anomaly_map)).clip(0,1))
            if t in t_harmonization:

                image_0 = res * orig_image + (1-res) * inputs   # orig_image, inputs

                image_0 = torch.clamp(image_0, 0, 1)
                image = self.train_scheduler.add_noise(original_samples=image_0, noise=noise, timesteps=t)

        self.img_ct += 1 
        image_refined = image
        if do_hmatching:
            image_refined = torch.Tensor(match_histograms(image.cpu().detach().numpy(), inputs.cpu().detach().numpy())).to(self.device)
            image_refined_ddpm = torch.Tensor(match_histograms(image_ddpm.cpu().detach().numpy(), inputs.cpu().detach().numpy())).to(self.device)
            intermediates_ddpm.append(image_refined_ddpm)
            intermediates.append(image_refined)
            intermediates_gt.append(inputs)

            res_thor = self.get_anomaly_mask(image_refined, inputs, hist_eq=do_hmatching)[loss_idx]
            res_ddpm = self.get_anomaly_mask(image_refined_ddpm, inputs, hist_eq=do_hmatching)[2]

            res_mix = resnp
            intermediates_res.append(res_mix)

            intermediates_res_ddpm.append(res_ddpm.cpu().detach().numpy())
        # tmp_intermediate_res = copy.deepcopy(intermediates_res)
        # np_inter_res = [inter.cpu().detach().numpy() for inter in tmp_intermediate_res]
        hmean = stats.hmean(np.stack(intermediates_res[:]), axis=0)
        # hmean_ddpm = stats.hmean(np.stack(intermediates_res_ddpm[:]), axis=0)
        hmean_mix = stats.hmean(np.stack(intermediates_res_mix[:]), axis=0)
        intermediates_res.append(hmean)
        # intermediates_res_ddpm.append(hmean_ddpm)
        intermediates_res_mix.append(hmean_mix)

        intermediates_res = [torch.Tensor(inter).to(self.device) for inter in intermediates_res]
        # intermediates_res_ddpm = [torch.Tensor(inter).to(self.device) for inter in intermediates_res_ddpm]
        intermediates_res_mix = [torch.Tensor(inter).to(self.device) for inter in intermediates_res_mix]

        intermediates_gt = intermediates
        intermediates_ddpm = intermediates
        intermediates_res_ddpm = intermediates_res
        if save_intermediates:
            return image_refined, {'z': intermediates, 'inter_gt': intermediates_gt, 'inter_res': intermediates_res, 'inter_ddpm': intermediates_ddpm, 'inter_res_ddpm': intermediates_res_ddpm, 
                           'inter_res_mix': intermediates_res_mix}
        else:
            return image_refined, {'z': None}

    @torch.no_grad()
    def get_likelihood(
        self,
        inputs: torch.Tensor,
        save_intermediates: bool | None = False,
        conditioning: torch.Tensor | None = None,
        original_input_range: tuple | None = (0, 255),
        scaled_input_range: tuple | None = (0, 1),
        verbose: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Computes the log-likelihoods for an input.
        Args:
            inputs: input images, NxCxHxW[xD]
            save_intermediates: save the intermediate spatial KL maps
            conditioning: Conditioning for network input.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
            verbose: if true, prints the progression bar of the sampling process.
        """

        if self.train_scheduler._get_name() != "DDPMScheduler":
            raise NotImplementedError(
                f"Likelihood computation is only compatible with DDPMScheduler,"
                f" you are using {self.train_scheduler._get_name()}"
            )
        if verbose and has_tqdm:
            progress_bar = tqdm(self.train_scheduler.timesteps)
        else:
            progress_bar = iter(self.train_scheduler.timesteps)
        intermediates = []
        total_kl = torch.zeros(inputs.shape[0]).to(inputs.device)
        for t in progress_bar:
            # Does this change things if we use different noise for every step?? before it was just one gaussian noise for all steps
            noise = generate_noise(self.train_scheduler.noise_type, inputs, t)

            timesteps = torch.full(
                inputs.shape[:1], t, device=inputs.device).long()
            noisy_image = self.train_scheduler.add_noise(
                original_samples=inputs, noise=noise, timesteps=timesteps)
            model_output = self.unet(
                x=noisy_image, timesteps=timesteps, context=conditioning)
            # get the model's predicted mean, and variance if it is predicted
            if model_output.shape[1] == inputs.shape[1] * 2 and self.train_scheduler.variance_type in ["learned", "learned_range"]:
                model_output, predicted_variance = torch.split(
                    model_output, inputs.shape[1], dim=1)
            else:
                predicted_variance = None

            # 1. compute alphas, betas
            alpha_prod_t = self.train_scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = self.train_scheduler.alphas_cumprod[t -
                                                                    1] if t > 0 else self.train_scheduler.one
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            # 2. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            if self.train_scheduler.prediction_type == "epsilon":
                pred_original_sample = (
                    noisy_image - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            elif self.train_scheduler.prediction_type == "sample":
                pred_original_sample = model_output
            elif self.train_scheduler.prediction_type == "v_prediction":
                pred_original_sample = (
                    alpha_prod_t**0.5) * noisy_image - (beta_prod_t**0.5) * model_output
            # 3. Clip "predicted x_0"
            if self.train_scheduler.clip_sample:
                pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

            # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_original_sample_coeff = (
                alpha_prod_t_prev ** (0.5) * self.train_scheduler.betas[t]) / beta_prod_t
            current_sample_coeff = self.train_scheduler.alphas[t] ** (
                0.5) * beta_prod_t_prev / beta_prod_t

            # 5. Compute predicted previous sample µ_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            predicted_mean = pred_original_sample_coeff * \
                pred_original_sample + current_sample_coeff * noisy_image

            # get the posterior mean and variance
            posterior_mean = self.train_scheduler._get_mean(
                timestep=t, x_0=inputs, x_t=noisy_image)
            posterior_variance = self.train_scheduler._get_variance(
                timestep=t, predicted_variance=predicted_variance)

            log_posterior_variance = torch.log(posterior_variance)
            log_predicted_variance = torch.log(
                predicted_variance) if predicted_variance else log_posterior_variance

            if t == 0:
                # compute -log p(x_0|x_1)
                kl = -self._get_decoder_log_likelihood(
                    inputs=inputs,
                    means=predicted_mean,
                    log_scales=0.5 * log_predicted_variance,
                    original_input_range=original_input_range,
                    scaled_input_range=scaled_input_range,
                )
            else:
                # compute kl between two normals
                kl = 0.5 * (
                    -1.0
                    + log_predicted_variance
                    - log_posterior_variance
                    + torch.exp(log_posterior_variance -
                                log_predicted_variance)
                    + ((posterior_mean - predicted_mean) ** 2) *
                    torch.exp(-log_predicted_variance)
                )
            total_kl += kl.view(kl.shape[0], -1).mean(axis=1)
            if save_intermediates:
                intermediates.append(kl.cpu())

        if save_intermediates:
            return total_kl, intermediates
        else:
            return total_kl

    def _approx_standard_normal_cdf(self, x):
        """
        A fast approximation of the cumulative distribution function of the
        standard normal. Code adapted from https://github.com/openai/improved-diffusion.
        """

        return 0.5 * (
            1.0 + torch.tanh(torch.sqrt(torch.Tensor([2.0 / math.pi]).to(
                x.device)) * (x + 0.044715 * torch.pow(x, 3)))
        )

    def _get_decoder_log_likelihood(
        self,
        inputs: torch.Tensor,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        original_input_range: tuple | None = (0, 255),
        scaled_input_range: tuple | None = (0, 1),
    ) -> torch.Tensor:
        """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image. Code adapted from https://github.com/openai/improved-diffusion.
        Args:
            input: the target images. It is assumed that this was uint8 values,
                        rescaled to the range [-1, 1].
            means: the Gaussian mean Tensor.
            log_scales: the Gaussian log stddev Tensor.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
        """
        assert inputs.shape == means.shape
        bin_width = (scaled_input_range[1] - scaled_input_range[0]) / (
            original_input_range[1] - original_input_range[0]
        )
        centered_x = inputs - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + bin_width / 2)
        cdf_plus = self._approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - bin_width / 2)
        cdf_min = self._approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            inputs < -0.999,
            log_cdf_plus,
            torch.where(inputs > 0.999, log_one_minus_cdf_min,
                        torch.log(cdf_delta.clamp(min=1e-12))),
        )
        assert log_probs.shape == inputs.shape
        return log_probs    