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
from torchvision.utils import save_image
import torch.nn.functional as F

#
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim as ssim2
from skimage import exposure
from scipy.ndimage.filters import gaussian_filter

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
        self.vgg_encoder = VGGEncoder().to(self.device)
        self.l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=True,
                                     lpips=True).to(self.device)
        self.l_cos = CosineSimLoss(device='cuda')
        self.l_ncc = NCC(win=[9, 9])

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
    

    def _log_visualization(self, to_visualize, i, count):
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

        wandb.log({f'Anomaly_masks/Example_Atlas_{count}': [wandb.Image(diffp, caption="Atlas_" + str(count))]})


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
                            self._log_visualization(to_visualize, i, count)

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

                to_visualize = [
                    {'title': 'x', 'tensor': x},
                    {'title': 'x_rec', 'tensor': x_rec},
                    {'title': f'Anomaly  map {anomaly_map.max():.3f}', 'tensor': anomaly_map, 'cmap': 'plasma',
                     'vmax': anomaly_map.max()},
                    {'title': 'gt', 'tensor': masks, 'cmap': 'plasma'}
                ]

                if 'mask' in x_rec_dict.keys():
                    masked_input = x_rec_dict['mask'] #+ x
                    masked_input[masked_input>1]=1

                    to_visualize.append({'title': 'Rec Orig', 'tensor': x_rec_dict['x_rec_orig'], 'cmap': 'gray'})
                    to_visualize.append({'title': 'Res Orig', 'tensor': x_rec_dict['x_res'], 'cmap': 'plasma',
                                        'vmax': x_rec_dict['x_res'].max()})
                    to_visualize.append({'title': 'Mask', 'tensor': masked_input, 'cmap': 'plasma', 'vmax': 0.5})

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
                            self._log_visualization(to_visualize, i, count)

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