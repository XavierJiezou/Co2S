<div align="center">

<img src="tools/docs/logo.png" alt="Co2S Logo" width="180" />

# Co2S

Toward Stable Semi-Supervised Remote Sensing Segmentation via Co-Guidance and Co-Fusion

[![arXiv](https://img.shields.io/badge/arXiv-2512.23035-b31b1b.svg)](https://arxiv.org/abs/2512.23035)
[![Project Page](https://img.shields.io/badge/Project%20Page-Co2S-blue)](https://xavierjiezou.github.io/Co2S/)
[![HuggingFace Models](https://img.shields.io/badge/🤗HuggingFace-Models-green)](https://huggingface.co/XavierJiezou/co2s-models)
[![HuggingFace Datasets](https://img.shields.io/badge/🤗HuggingFace-Datasets-orange)](https://huggingface.co/datasets/XavierJiezou/co2s-datasets)
[![HuggingFace Papers](https://img.shields.io/badge/🤗%20HuggingFace-Papers-yellow)](https://huggingface.co/papers/2512.23035)
<!--[![Overleaf](https://img.shields.io/badge/Overleaf-Open-green?logo=Overleaf&style=flat)](https://www.overleaf.com/project/6695fd4634d7fee5d0b838e5)-->

<!--Love the project? Please consider [donating](https://paypal.me/xavierjiezou?country.x=C2&locale.x=zh_XC) to help it improve!-->

![framework](tools/docs/Co2S.png)

</div>

<!--This repository serves as the official implementation of the paper **"Adapting Vision Foundation Models for Robust Cloud Segmentation in Remote Sensing Images"**. It provides a comprehensive pipeline for semantic segmentation, including data preprocessing, model training, evaluation, and deployment, specifically tailored for cloud segmentation tasks in remote sensing imagery.-->

---



## Installation

1. Clone the Repository

```bash
git clone https://github.com/XavierJiezou/Co2S.git
cd Co2S
```

2. Install Dependencies

You can either set up the environment manually or use our pre-configured environment for convenience:

- Option 1: Manual Installation

```bash
conda create -n Co2S python=3.8.20
conda activate Co2S
pip install -r requirements.txt
```

- Option 2: Use Pre-configured Environment

We provide a pre-configured environment (`env.tar.gz`) hosted on Hugging Face. You can download it directly from [Hugging Face](https://huggingface.co/XavierJiezou/co2s-models/blob/main/env.tar.gz). Follow the instructions on the page to set up and activate the environment.

You can use the following command to download the package:

```bash
wget -O env.tar.gz https://huggingface.co/XavierJiezou/co2s-models/resolve/main/env.tar.gz
```

Once download env.tar.gz, you can extract it using the following command:

```bash
tar -xzf env.tar.gz -C envs
source envs/bin/activate
conda-unpack
```


## Prepare Data  

We have open-sourced all datasets used in the paper, which are hosted on [Hugging Face Datasets](https://huggingface.co/datasets/XavierJiezou/co2s-datasets). Please follow the instructions on the dataset page to download the data.  

You can use the following script to download the specific datasets and automatically rename the folders to lowercase:

```bash
pip install -U "huggingface_hub[cli]"

# Download and rename datasets (GID, LOVEDA, MER, MSL, POTSDAM, WHDLD) to lowercase
for DATASET in GID LOVEDA MER MSL POTSDAM WHDLD; do
  echo "Downloading $DATASET..."
  hf download XavierJiezou/co2s-datasets --repo-type dataset --include "$DATASET/*" --local-dir data
  # Rename directory to lowercase (e.g., data/GID -> data/gid)
  mv "data/$DATASET" "data/$(echo $DATASET | tr '[:upper:]' '[:lower:]')"
done
```

After downloading, organize the dataset as follows:  

```  
Co2S
├── ...
├── data
│   ├── whdld
│   │   ├── images
│   │   ├── labels
│   ├── potsdam
│   │   ├── images
│   │   ├── labels
│   ├── loveda
│   │   ├── images
│   │   ├── labels
│   ├── gid
│   │   ├── images
│   │   ├── labels
│   ├── mer
│   │   ├── images
│   │   ├── labels
│   ├── msl
│   │   ├── images
│   │   ├── labels
```   



## Training

### Step 1:  Download pretrained weights of vision foundation models

We provide the pre-trained weights (including converted CLIP and DINOv3) directly on [Hugging Face](https://huggingface.co/XavierJiezou/co2s-models/tree/main/pretrained). You can download them to the pretrained/ directory using the following command:

```bash
# Download the 'pretrained' folder content to the local current directory
hf download XavierJiezou/co2s-models --include "pretrained/*" --local-dir .
```

After downloading, Please extract the Pre-Trained Backbones to the following folder structure:
```
Co2S
├── ...
├── pretrained
│   ├── clip2mmseg_ViT16_clip_backbone.pth
│   ├── dinov3_vitb16_pretrain_lvd1689m.pth
```

### Step 2: Modify the Configuration File

After converting the backbone network weights, make sure to correctly specify the path to the configuration file within your config settings.

For example: 
```bash
nano configs/_base_/models/dual_model_clip.py
```

```python
# configs/_base_/models/dual_model_clip.py
model = dict(
    type='Co2S',
    pretrained='pretrained/clip2mmseg_ViT16_clip_backbone.pth', # you can set weight path here
    backbone=dict(
    ...
    ),
)
```

Update the `configs` directory with your training configuration, or use one of the provided example configurations. You can customize the backbone, dataset paths, and hyperparameters in the configuration file.  

### Step 3: Start Training  

Use the following command to begin training:  

```bash  
python experiments.py --exp EXP_ID --run RUN_ID
#  e.g. EXP_ID=40; RUN_ID=0 for Co2S on WHDLD with 1/24 labels
#  WHDLD exp_id == 40: splits = ['1_24', '1_16', '1_8', '1_4']
#  LoveDA exp_id == 41: splits = ['1_40', '1_16', '1_8', '1_4']
#  Potsdam exp_id == 42: splits = ['1_32', '1_16', '1_8', '1_4']
#  GID exp_id == 43: splits = ['1_8', '1_4']
#  MER exp_id == 44: splits = ['1_8', '1_4']
#  MSL exp_id == 45: splits = ['1_8', '1_4']
```  

## Evaluation

All model weights used in the paper have been open-sourced and are available on [Hugging Face Models](https://huggingface.co/XavierJiezou/co2s-models).

You can use the following script to download the checkpoints for the ablation studies and all dataset experiments (GID, POTSDAM, etc.) into the exp/ directory:

```bash
# Download Ablation_Experiment and dataset checkpoints (GID, LOVEDA, etc.) to the local 'exp' folder
for DIR in Ablation_Experiment GID LOVEDA MER MSL POTSDAM WHDLD; do
  echo "Downloading $DIR..."
  huggingface-cli download XavierJiezou/co2s-models --include "$DIR/*" --local-dir exp
done
```

Use the following command to evaluate the trained model:  

```bash  
python -m third_party.unimatch.eval \
    --config PATH/TO/CONFIG \                 # e.g., exp/POTSDAM/1_32-74.30/config.yaml
    --save-path PATH/TO/CHECKPOINT_DIR \      # e.g., exp/POTSDAM/1_32-74.30/
    --pred-path PATH/TO/PREDICTION_OUTPUT     # e.g., POTSDAM/1_32/
```  

## Inference

Use the following command to infer the trained model:  

```bash
python inference.py \
  --config PATH/TO/CONFIG \          # e.g., exp/POTSDAM/1_32-74.30/config.yaml
  --model MODEL_NAME \               # e.g., clip (or dinov3)
  --checkpoint PATH/TO/CHECKPOINT \  # e.g., exp/POTSDAM/1_32-74.30/best_clip.pth
  --pred-path PATH/TO/PRED_LIST \    # e.g., splits/potsdam/pred.txt
  --output-dir PATH/TO/OUTPUT        # e.g., output/POTSDAM/1_32/
```

## Visualization

We have published the pre-trained model's visualization results of various datasets on Hugging Face at [Hugging Face](https://huggingface.co/datasets/XavierJiezou/co2s-datasets/tree/main/Visualization_results). If you prefer not to run the code, you can directly visit the repository to download the visualization results. 

Alternatively, you can download the results directly to your local machine using the following command:

```bash
# Download the 'Visualization_results' folder from the dataset repository
huggingface-cli download XavierJiezou/co2s-datasets --repo-type dataset --include "Visualization_results/*" --local-dir .
```

## Attention Map

We provide specific tools to visualize the 12 individual attention heads, as well as the Average and Max attention projections for both CLIP and DINOv3 models. This helps in understanding the global vs. local feature focus of the backbone networks.

Use the following command to generate attention maps:

```bash
python tools/clip_dinov3_attention_map/attention_map.py \
  --image-path PATH/TO/IMAGE \       # e.g., tools/clip_dinov3_attention_map/A.png
  --model MODEL_NAME \               # e.g., clip (or dinov3)
  --checkpoint PATH/TO/CHECKPOINT \  # e.g., pretrained/clip2mmseg_ViT16_clip_backbone.pth
  --output-dir PATH/TO/OUTPUT        # e.g., tools/clip_dinov3_attention_map/clip_attention/
```

The results will then include individual head images (.png) and a complete summary table (.pdf), which will be saved in the specified output directory.

The sample visualization results are as follows:

<table>
  <tr>
    <td align="center" width="50%">
      <img src="tools/clip_dinov3_attention_map/clip_summary_grid.svg" alt="Figure 1" style="width: 100%;">
      <br>
      CLIP Attention Maps
    </td>
    <td align="center" width="50%">
      <img src="tools/clip_dinov3_attention_map/dinov3_summary_grid.svg" alt="Figure 2" style="width: 100%;">
      <br>
      DINOv3 Attention Maps
    </td>
  </tr>
</table>
