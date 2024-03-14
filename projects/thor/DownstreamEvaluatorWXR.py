import logging
from torch.nn import L1Loss
import copy
#
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import wandb
import plotly.graph_objects as go
import seaborn as sns
# import umap.umap_ as umap
from dl_utils import data_utils as du
#
from skimage.metrics import structural_similarity as ssim

#
import lpips
from model_zoo.vgg import VGGEncoder
#
from core.DownstreamEvaluator import DownstreamEvaluator
#
from transforms.synthetic import *


class PDownstreamEvaluator(DownstreamEvaluator):
    """
    Federated Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, name, model, device, test_data_dict, checkpoint_path, global_=True):
        super(PDownstreamEvaluator, self).__init__(name, model, device, test_data_dict, checkpoint_path)

        self.criterion_rec = L1Loss().to(self.device)
        self.compute_scores = True
        self.vgg_encoder = VGGEncoder().to(self.device)
        self.l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=True, lpips=True).to(self.device)

    def start_task(self, global_model):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
                   the model weights
        """
        # th = 0.00113 # AutoDDPM
        # th = 0.119 # AnoDDPM
        th = 0.08# THOR Gaussian
        # th = 0.081 # THOR Simplex
        # th = 0.223 # DDPM 
        # th = 0.171 # DDPM-Gaussian
        # _ = self.thresholding(global_model)
        self.object_localization(global_model, th)


    def thresholding(self, global_model):
        """
                Validation of downstream tasks
                Logs results to wandb

                :param global_model:
                    Global parameters
                """
        logging.info("################ Threshold Search #################")
        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()
        ths = np.linspace(0, 1, endpoint=False, num=1000)
        fprs = dict()
        for th_ in ths:
            fprs[th_] = []
        im_scale = 128 * 128
        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            logging.info('DATASET: {}'.format(dataset_key))
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                    filenames = data[2]
                x = data0.to(self.device)
                # print(filenames)
                anomaly_maps, anomaly_scores, x_rec_dict = self.model.get_anomaly(x)
                for i in range(len(x)):
                    x_res_i = anomaly_maps[i][0]   # default
                    # x_res_i *= 100
                    for th_ in ths:
                        fpr = (np.count_nonzero(x_res_i > th_) * 100) / im_scale
                        fprs[th_].append(fpr)
        mean_fprs = []
        for th_ in ths:
            mean_fprs.append(np.mean(fprs[th_]))
        mean_fprs = np.asarray(mean_fprs)
        sorted_idxs = np.argsort(mean_fprs)
        th_1, th_2, best_th = 0, 0, 0
        fpr_1, fpr_2, best_fpr = 0, 0, 0
        for sort_idx in sorted_idxs:
            th_ = ths[sort_idx]
            fpr_ = mean_fprs[sort_idx]
            if fpr_ <= 1:
                th_1 = th_
                fpr_1 = fpr_
            if fpr_ <= 2:
                th_2 = th_
                fpr_2 = fpr_
            if fpr_ <= 5:
                best_th = th_
                best_fpr = fpr_
            else:
                break
        print(f'Th_1: [{th_1}]: {fpr_1} || Th_2: [{th_2}]: {fpr_2} || Th_5: [{best_th}]: {best_fpr}')
        return best_th

    def object_localization(self, global_model, th=0):
        """
        Validation of downstream tasks
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        print(f'OBJECT LOCALIZATION USING TH: {th}')
        anomaly_list = [
            # Bone Anomaly
            "./data/wrist/pediatric/5797_0233713488_03_WRI-L1_M015.png",
            "./data/wrist/pediatric/5509_0695010795_01_WRI-R1_M018.png",
            "./data/wrist/pediatric/4407_1097755929_01_WRI-R1_M016.png",
            "./data/wrist/pediatric/4730_0363979493_01_WRI-R1_F014.png",
            "./data/wrist/pediatric/0602_0247084664_01_WRI-L1_F016.png",
            "./data/wrist/pediatric/4936_0673700714_04_WRI-L1_F013.png",
            "./data/wrist/pediatric/3577_1038087132_03_WRI-L1_F003.png",
            "./data/wrist/pediatric/3231_1061493919_01_WRI-R1_F017.png",

            # Foreign Body
            "./data/wrist/pediatric/1237_0692065553_01_WRI-L1_M007.png",

            # Fractures
            "./data/wrist/pediatric/2898_0832274444_03_WRI-R1_M010.png",
            "./data/wrist/pediatric/3335_0914852727_03_WRI-L1_F006.png",
            "./data/wrist/pediatric/3385_0852136661_04_WRI-L1_F008.png",
            "./data/wrist/pediatric/3594_0331667659_01_WRI-R1_M014.png",
            "./data/wrist/pediatric/3662_0879753291_04_WRI-L1_M015.png",
            "./data/wrist/pediatric/3690_0715622775_01_WRI-L1_F004.png",
            "./data/wrist/pediatric/3779_0706680034_02_WRI-L1_M010.png",
            "./data/wrist/pediatric/3786_0681462591_01_WRI-R1_M017.png",
            "./data/wrist/pediatric/3799_0950453301_04_WRI-L1_M010.png",
            "./data/wrist/pediatric/3833_0229713858_07_WRI-L1_F006.png",
            "./data/wrist/pediatric/3864_1047054876_03_WRI-L1_M012.png",
            "./data/wrist/pediatric/4007_0842579452_01_WRI-R1_F001.png",
            "./data/wrist/pediatric/4072_0382348481_06_WRI-L1_M006.png",
            "./data/wrist/pediatric/4089_0162050042_05_WRI-L1_F007.png",
            "./data/wrist/pediatric/4185_0991015525_04_WRI-R1_F008.png",
            "./data/wrist/pediatric/4274_0536924068_02_WRI-L1_M013.png",
            "./data/wrist/pediatric/4341_0971012016_01_WRI-L1_M010.png",
            "./data/wrist/pediatric/4365_0570097034_04_WRI-R1_F010.png",
            "./data/wrist/pediatric/4408_0721644151_01_WRI-R1_F013.png",
            "./data/wrist/pediatric/4421_1101980183_03_WRI-R1_M009.png",
            "./data/wrist/pediatric/4488_0420504559_05_WRI-L1_M014.png",
            "./data/wrist/pediatric/4605_0742862296_03_WRI-L1_F011.png",
            "./data/wrist/pediatric/4610_0629393281_08_WRI-L1_M011.png",
            "./data/wrist/pediatric/4639_1072961779_04_WRI-L1_F009.png",
            "./data/wrist/pediatric/4663_0619072593_02_WRI-R1_M015.png",
            "./data/wrist/pediatric/4672_1041752926_06_WRI-L1_F010.png",
            "./data/wrist/pediatric/4695_0371504130_01_WRI-L1_F009.png",
            "./data/wrist/pediatric/4719_1120087077_04_WRI-R1_F005.png",
            "./data/wrist/pediatric/4779_1160661927_01_WRI-R1_M005.png",
            "./data/wrist/pediatric/4787_1174908641_01_WRI-R1_M013.png",
            "./data/wrist/pediatric/4844_0314423840_01_WRI-L1_M011.png",
            "./data/wrist/pediatric/4857_1142512150_03_WRI-L1_M014.png",
            "./data/wrist/pediatric/4893_0183062895_02_WRI-L1_M008.png",
            "./data/wrist/pediatric/4920_0883517728_01_WRI-L1_M013.png",
            "./data/wrist/pediatric/4933_1287513487_01_WRI-L1_F012.png",
            "./data/wrist/pediatric/4951_0419405351_01_WRI-R1_M014.png",
            "./data/wrist/pediatric/5060_1106103142_01_WRI-L1_F007.png",
            # Metal
            "./data/wrist/pediatric/5048_0683614110_01_WRI-L1_F014.png",
            "./data/wrist/pediatric/3963_1041916471_04_WRI-L1_M017.png",
            "./data/wrist/pediatric/1780_0522712705_05_WRI-R1_M017.png",
            "./data/wrist/pediatric/5795_0739161695_05_WRI-L1_M005.png",
            "./data/wrist/pediatric/2059_0960888711_03_WRI-R1_F008.png",
            "./data/wrist/pediatric/2376_0912930811_02_WRI-R1_M014.png",
            "./data/wrist/pediatric/0133_0282914262_06_WRI-R1_M014.png",
            "./data/wrist/pediatric/3963_1041916471_04_WRI-L1_M017.png",
            "./data/wrist/pediatric/3799_0949300875_03_WRI-L1_M010.png"
        ]

        logging.info("################ Object Localzation TEST #################" + str(th))
        lpips_alex = lpips.LPIPS(net='alex')  # best forward scores
        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()
        metrics = {
            'MSE': [],
            'LPIPS': [],
            'SSIM': [],
            'TP': [],
            'FP': [],
            'Precision': [],
            'Recall': [],
            'F1': [],
        }
        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'MSE': [],
                'LPIPS': [],
                'SSIM': [],
                'TP': [],
                'FP': [],
                'Precision': [],
                'Recall': [],
                'F1': [],
            }
            logging.info('DATASET: {}'.format(dataset_key))
            tps, fns, fps = 0, 0, []
            for idx, data in enumerate(dataset):
                if type(data) is dict and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                    data1 = data[1].shape
                x = data0.to(self.device)
                masks_bool = True if len(data1) > 2 else False
                nr_batches, nr_slices, width, height = data0.shape
                # x = data0.view(nr_batches * nr_slices, 1, width, height)
                x = data0.to(self.device)
                masks = data[1][:, 0, :, :].view(nr_batches, 1, width, height).to(self.device)\
                    if masks_bool else None
                neg_masks = data[1][:, 1, :, :].view(nr_batches, 1, width, height).to(self.device)
                neg_masks[neg_masks>0.5] = 1
                neg_masks[neg_masks<1] = 0
                filenames = data[2]

                # print(f'mask bool: {masks_bool}, min : {torch.min(masks)}, max: {torch.max(masks)}, avg:'
                #       f' {torch.mean(masks)}, shape: {masks.shape}')
                # print(f'Neg: min : {torch.min(neg_masks)}, max: {torch.max(neg_masks)}, avg: {torch.mean(neg_masks)}, shape: {neg_masks.shape}')
                # cast_files = du.get_data_from_csv('./data/wrist/splits/test_cast.csv')
                # if filename in cast_files:
                #     print(f'Skipping cast file: {filename}')
                #     continue
                filename = filenames[0]

                # if filename not in anomaly_list:
                    # continue

                anomaly_maps, anomaly_scores, x_rec_dict = self.model.get_anomaly(x)
                x_rec = x_rec_dict['x_rec']
                for i in range(len(x)):
                    count = str(idx * len(x) + i)
                    if int(count) != 20: 
                        continue
                    x_i = x[i][0]
                    x_rec_i = x_rec[i][0]
                    x_res_i = anomaly_maps[i][0] # default
                    # print(x_res_i.shape)

                    mask_ = masks[i][0].cpu().detach().numpy() if masks_bool else None
                    neg_mask_ = neg_masks[i][0].cpu().detach().numpy() if masks_bool else None
                    bboxes = cv2.cvtColor(neg_mask_*255, cv2.COLOR_GRAY2RGB)
                    # thresh_gt = cv2.threshold((mask_*255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    cnts_gt = cv2.findContours((mask_*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts_gt = cnts_gt[0] if len(cnts_gt) == 2 else cnts_gt[1]
                    gt_box = []
                    for c_gt in cnts_gt:
                        x, y, w, h = cv2.boundingRect(c_gt)
                        gt_box.append([x, y, x+w, y+h])
                        cv2.rectangle(bboxes, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    #
                    # loss_mse = self.criterion_rec(x_rec_i, x_i)
                    # test_metrics['MSE'].append(loss_mse.item())
                    # loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), x_rec_i.cpu()).detach().numpy())
                    # test_metrics['LPIPS'].append(loss_lpips)
                    #
                    x_ = x_i.cpu().detach().numpy()
                    x_rec_ = x_rec_i.cpu().detach().numpy()
                    # np.save(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_rec.npy', x_rec_)

                    # ssim_ = ssim(x_rec_, x_, data_range=1.)
                    # test_metrics['SSIM'].append(ssim_)

                    x_combo = copy.deepcopy(x_res_i)
                    x_combo[x_combo < th] = 0

                    x_pos = x_combo * mask_
                    x_neg = x_combo * neg_mask_
                    res_anomaly = np.sum(x_pos)
                    res_healthy = np.sum(x_neg)
                    # print(np.sum(x_neg), np.sum(x_pos))

                    amount_anomaly = np.count_nonzero(x_pos)
                    amount_mask = np.count_nonzero(mask_)

                    tp = 1 if amount_anomaly > 0.1 * amount_mask else 0
                    tps += tp
                    fn = 1 if tp == 0 else 0
                    fns += fn

                    fp = min(10, int(res_healthy / max(res_anomaly,1))) #[i for i in ious if i < 0.1]
                    fps.append(fp)
                    precision = tp / max((tp+fp), 1)
                    test_metrics['TP'].append(tp)
                    test_metrics['FP'].append(fp)
                    test_metrics['Precision'].append(precision)
                    test_metrics['Recall'].append(tp)
                    test_metrics['F1'].append(2 * (precision * tp) / (precision + tp + 1e-8))

                    ious = [res_anomaly, res_healthy]
                    if filename in anomaly_list or int(count) in [0,1,2,3,10,15,20,30]:
                        # bone lesions
                    # if True:
                    # if (idx % 20) == 0: # and (i % 5 == 0) or int(count)==13600 or int(count)==40:
                        elements = [x_, x_rec_, x_res_i, x_combo]
                        v_max_res =  np.percentile(x_res_i, 99.5)
                        v_maxs = [1, 1,v_max_res,v_max_res]
                        titles = [filename, 'Rec', 'Res', '5%FPR']
                        # if 'embeddings' in x_rec_dict.keys():
                        # #     coarse_y = x_rec_dict['y_coarse'][i][0].cpu().detach().numpy()
                        # #     masked_x = x_rec_dict['masked'][i][0].cpu().detach().numpy()
                        # #     x_coarse_res = x_rec_dict['residual'][i][0]
                        # #     saliency_coarse = x_rec_dict['saliency'][i][0]
                        #     elements = [x_, coarse_y, x_coarse_res, saliency_coarse, masked_x, x_rec_, x_,
                        #                 x_res_orig, x_res_orig * x_coarse_res, saliency_i, saliency_i * saliency_coarse,
                        #                 x_res_i] #np.max(x_coarse_res)
                        #     v_maxs = [1, 1, 0.5, np.max(saliency_coarse), 1, 1, 1, np.max(x_res_orig),
                        #               np.max(x_res_orig * x_coarse_res), np.max(saliency_i), np.max(saliency_i * saliency_coarse), 0.25]#np.max(x_res_i)]
                        #     titles = ['Input', 'C_Rec', str(np.max(x_coarse_res)), str(np.max(saliency_coarse)), 'Masked', 'Rec', 'Input',
                        #               str(np.max(x_res_orig)), str(np.max(x_res_orig * x_coarse_res)),
                        #               str(np.max(saliency_i)), str(np.max(saliency_i*saliency_coarse)),
                        #               str(np.max(x_res_i))]

                        if masks_bool:
                            # elements.append(saliency_vgg)
                            # elements.append(saliency_alex)
                            # elements.append(ra_ssim)
                            elements.append(bboxes.astype(np.int64))
                            elements.append(x_pos)
                            elements.append(x_neg)
                            # v_maxs.append(0.99)
                            # v_maxs.append(0.99)
                            v_maxs.append(1)
                            v_maxs.append(v_max_res)
                            v_maxs.append(v_max_res)
                            # titles.append('VGG')
                            # titles.append('Alex')
                            titles.append('GT')
                            titles.append(str(np.round(res_anomaly, 2)) + ', TP: ' + str(tp))
                            titles.append(str(np.round(res_healthy, 2)) + ', FP: ' + str(fp))
                        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                        diffp.set_size_inches(len(elements) * 4, 4)
                        for idx_arr in range(len(axarr)):
                            axarr[idx_arr].axis('off')
                            v_max = v_maxs[idx_arr]
                            c_map = 'gray' if v_max == 1 else 'plasma'
                            axarr[idx_arr].imshow(np.squeeze(elements[idx_arr]), vmin=0, vmax=v_max, cmap=c_map)
                            axarr[idx_arr].set_title(titles[idx_arr])

                            wandb.log({'Anomaly/Example_' + dataset_key + '_' + str(count): [
                                wandb.Image(diffp, caption="Sample_" + str(count) + filename)]})


            # pred_dict[dataset_key] = (precisions, recalls)
            # fps_ = len(np.unique(np.asarray(fps)))
            # precision = tps / (tps+fps_+ 1e-8)
            # recall = tps/(tps+fns)
            # precision = np.mean(precisions)
            # recall = np.mean(recalls)
            # logging.info(f' TP: {tps}, FN:  {fns}, FP: {fps_}')
            # logging.info(f' Precision: {precision}; Recall: {recall}')
            # logging.info(f' F1: {2 * (precision * recall) / (precision + recall+ 1e-8)}')

            for metric in test_metrics:
                logging.info('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                                         np.nanstd(test_metrics[metric])))
                if metric == 'TP':
                    logging.info(f'TP: {np.sum(test_metrics[metric])} of {len(test_metrics[metric])} detected')
                if metric == 'FP':
                    logging.info(f'FP: {np.sum(test_metrics[metric])} missed')
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

    def umap_plot(self, global_model):
        compute_umap = True
        plot_saliency = False
        """
                Validation of downstream tasks
                Logs results to wandb

                :param global_model:
                    Global parameters
                """
        logging.info("################ TSNE TEST #################")
        self.model.load_state_dict(global_model)
        self.model.eval()
        z = None
        z_ = None
        labels = []
        labels_ = []
        idx_global = 0
        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            logging.info('DATASET: {}'.format(dataset_key))
            for idx, data in enumerate(dataset):
                count = 0
                if 'dict' in str(type(data)) and 'images' in data.keys():
                    data0 = data['images']
                    data1 = [0]
                else:
                    data0 = data[0]
                    data1 = data[1].shape
                # print(data[1].shape)
                nr_batches, nr_slices, width, height = data0.shape
                # x = data0.view(nr_batches * nr_slices, 1, width, height)
                x = data0.to(self.device)
                x_rec, x_rec_dict = self.model(x, deterministic=False)
                embedding = x_rec_dict['z']
                # anomaly_input = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
                # embedding = self.vgg_encoder(anomaly_input)[2]
                if compute_umap:
                    z = torch.flatten(embedding, start_dim=1).cpu().detach().numpy() if z is None else \
                        np.concatenate((z, torch.flatten(embedding, start_dim=1).cpu().detach().numpy()), axis=0)
                    for i in range(len(embedding)):
                        labels.append(dataset_key)
                x_rec_rec, x_rec_rec_dict = self.model(x_rec.detach())
                # ph_input = x_rec.repeat(1, 3, 1, 1) if x_rec.shape[1] == 1 else x_rec
                # embedding_ph = self.vgg_encoder(ph_input)[2]
                embedding_ph = x_rec_rec_dict['z']
                if compute_umap:
                    z = torch.flatten(embedding_ph, start_dim=1).cpu().detach().numpy() if z is None else \
                        np.concatenate((z, torch.flatten(embedding_ph, start_dim=1).cpu().detach().numpy()), axis=0)
                    for i in range(len(embedding_ph)):
                        labels.append(dataset_key + '_PH')
                # if compute_umap:
                #     z_ = torch.flatten(torch.abs(embedding_ph-embedding), start_dim=1).cpu().detach().numpy() if z is None else \
                #         np.concatenate((z, torch.flatten(torch.abs(embedding_ph-embedding), start_dim=1).cpu().detach().numpy()), axis=0)
                #     labels_.append(dataset_key)
                # if plot_saliency:
                #     saliency = gaussian_filter(self.lpips_loss(x_rec.detach(), x.detach()), sigma=2)
                #     # saliency_vgg = self.compute_perceptual_anomaly(x[0][0].detach(), x_rec[0][0].detach())
                #     # saliency = self.embedding_loss_ra(self.model, x_rec, x)
                #     x_ = x.detach().cpu().numpy()
                #     x_rec_ = x_rec.detach().cpu().numpy()
                #     x_res = np.abs(x_rec_ - x_)
                #     combo = x_res * saliency
                #
                elements = [x.cpu().detach().numpy(), x_rec.cpu().detach().numpy()]
                v_maxs = [1, 1]
                titles = ['Input_' + str(idx_global), 'Rec_' + str(idx_global+1)]

                diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                diffp.set_size_inches(len(elements) * 4, 4)
                for idx_arr in range(len(axarr)):
                    axarr[idx_arr].axis('off')
                    v_max = v_maxs[idx_arr]
                    c_map = 'gray' if v_max == 1 else 'plasma'
                    axarr[idx_arr].imshow(np.squeeze(elements[idx_arr]), vmin=0, vmax=v_max, cmap=c_map)
                    axarr[idx_arr].set_title(titles[idx_arr])

                    wandb.log({'Anomaly/Example_' + dataset_key + '_' + str(idx_global): [
                        wandb.Image(diffp, caption="Sample_" + str(idx_global))]})

                idx_global += 1

        if compute_umap:
            z = np.asarray(z)
            labels = np.asarray(labels)
            reducer = umap.UMAP(min_dist=0.6, n_neighbors=25, metric='euclidean', init='random')
            umap_dim = reducer.fit_transform(z)

            for idx, label  in  enumerate(labels):
                print(f' {str(idx)}: ({label}),  [{str(umap_dim[idx, 1])}, {str(umap_dim[idx, 0])}]')
            sns.set_style("whitegrid")

            # sns.set(context='paper', style='darkgrid', rc={'figure.facecolor':'white'}, font_scale=1.2)
            colors = ['#acce81', '#8c9fc5', '#d5121b', '#4766a2', '#ff7373', '#76b6b6', '#800000', '#468499']
            # colors = ['#acce81', '#d5121b', '#ff7373', '#800000']

            # fig = plt.figure()
            # sns.color_palette(colors)
            fig, ax = plt.subplots()
            sns_plot = sns.jointplot(x=umap_dim[:, 1], y=umap_dim[:, 0], hue=labels, palette=sns.color_palette(colors),
                                     s=40)
            sns_plot.ax_joint.legend(loc='center right', bbox_to_anchor=(-0.2, 0.5))
            sns_plot.savefig(self.checkpoint_path + "/output_umap.pdf")
            wandb.log({"umap_plot": fig})
            logging.info('DONE')
            # wandb.log({"umap_image": [wandb.Image(sns_plot, caption="UMAP_Image")]})

            # sns_plot = sns.jointplot(x=tsne[:,1], y=tsne[:,0], hue=labels, palette="deep", s=50)

