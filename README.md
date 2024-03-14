# THOR_DDPM
<p align="center">
<img src="https://github.com/ci-ber/THOR_DDPM/assets/106509806/ccdaedc6-458a-4e5b-9884-f209e780af1e" width=200>
</p>

<h1 align="center">
  <br>
Diffusion Models with Implicit Guidance for Medical Anomaly Detection
  <br>
</h1>
</h1>
  <p align="center">
    <a href="https://cosmin-bercea.com">Cosmin Bercea</a> •
    <a href="https://www.neurokopfzentrum.med.tum.de/neuroradiologie/mitarbeiter-profil-wiestler.html">Benedikt Wiestler</a> •
    <a href="https://aim-lab.io/author/daniel-ruckert/">Daniel Rueckert </a> •
    <a href="https://compai-lab.github.io/author/julia-a.-schnabel/">Julia A. Schnabel </a>
  </p>
<h4 align="center">Official repository of the paper</h4>
<a href="https://arxiv.org/pdf/2403.08464.pdf">Preprint</a> </h4>

<p align="center">
<img src="https://github.com/ci-ber/THOR_DDPM/assets/106509806/09f646e7-944a-4be1-bf15-d067c72954b8">
</p>

## Citation

If you find our work useful, please cite our paper:
```
@misc{Bercea2024diffusion,
    title={Diffusion Models with Implicit Guidance for Medical Anomaly Detection},
    author={Cosmin I. Bercea and Benedikt Wiestler and Daniel Rueckert and Julia Schnabel},
    year={2024},
    month={3},
    eprint={2403.08464},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

> **Abstract:** *Diffusion models have advanced unsupervised anomaly detection by improving the transformation of pathological images into pseudo-healthy equivalents. Nonetheless, standard approaches may compromise critical information during pathology removal, leading to restorations that do not align with unaffected regions in the original scans. Such discrepancies can inadvertently increase false positive rates and reduce specificity, complicating radiological evaluations. This paper introduces Temporal Harmonization for Optimal Restoration (THOR), which refines the de-noising process by integrating implicit guidance through temporal anomaly maps. THOR aims to preserve the integrity of healthy tissue in areas unaffected by pathology. Comparative evaluations show that THOR surpasses existing diffusion-based methods in detecting and segmenting anomalies in brain MRIs and wrist X-rays. Code: https://github.com/ci-ber/THOR_DDPM.*


## Setup and Run

The code is based on the deep learning framework from the Institute of Machine Learning in Biomedical Imaging: https://github.com/compai-lab/iml-dl

### Framework Overview 

<p align="center">
<img src="https://github.com/ci-ber/THOR_DDPM/assets/106509806/87c17548-fd15-4c9d-80cf-ecaa11cc7f61">
</p>

#### 1). Clone repository

```bash
git clone https://github.com/ci-ber/THOR_DDPM.git
cd THOR_DDPM
```

#### 2). Install PyTorch 

*Optional* create virtual env:
```bash
conda create --name thor python=3.8.0
conda activate thor
```

> Example installation: 
* *with cuda*: 
```
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
* *w/o cuda*:
```
pip3 install torch==1.9.1 torchvision==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 3). Install requirements

```bash
pip install -r pip_requirements.txt
```

#### 4). Set up wandb (https://docs.wandb.ai/quickstart)

Sign up for a free account and login to your wandb account.
```bash
wandb login
```
Paste the API key from https://wandb.ai/authorize when prompted.

#### 5). Download datasets 

<h4 align="center"><a href="https://brain-development.org/ixi-dataset/">IXI</a> • <a href="https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html">Atlas (Stroke) </a> • <a href="https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193">Pediatric Wrist X-ray </a> </h4>

> Move the datasets to the target locations. You can find detailed information about the expected files and locations in the corresponding *.csv files under data/$DATASET/splits.

> *Alternatively you can use your own mid-axial slices of T1w brain scans with our pre-trained weights for <a href="https://www.dropbox.com/scl/fi/55cl3821vw1jp3jim2da2/brain_Gaussian.pt?rlkey=pz99o0x3g6vi3siwtvfpb0oyo&dl=1"> Gaussian noise</a> or <a href="https://www.dropbox.com/scl/fi/d8olm81iynd4lbsjt0fgm/brain_Simplex.pt?rlkey=onmyjogb3ej7uibs7r4poy4w8&dl=1"> Simplex noise</a> or train from scratch on other anatomies and modalities.*

> * For pediatric wrist X-rays you can use our pre-trained weights for <a href="https://www.dropbox.com/scl/fi/dd0zzzjcimmw3egcfvhvu/wxr_Gaussian.pt?rlkey=iovq4hx9zcmlogszhg19d9ou2&dl=1"> Gaussian noise</a> or <a href="https://www.dropbox.com/scl/fi/0aeiawcih2io4imdo169f/wxr_Simplex.pt?rlkey=grn8t62nsn0ojo6rc378gemvt&dl=1"> Simplex noise</a>. 

#### 6). Run the pipeline

Run the main script with the corresponding config like this:

```bash
python core/Main.py --config_path ./projects/thor/configs/brain/thor.yaml
```

Refer to the thor.yaml for the default configuration.

# That's it, enjoy! :rocket:
