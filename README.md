# Introduction

Zero-shot 3D anomaly detection aims to identify and localize anomalies in unseen object categories without accessing their training samples. However, existing methods often rely on a unified anomaly representation, making it difficult to distinguish global structural changes from local fine-grained defects. We propose GHP-CLIP, a geometry-guided hierarchical prompt framework for zero-shot 3D anomaly detection. GHP- CLIP extracts global geometric and local defect features from point clouds to construct complementary parent- and child- level prompts. We further introduce Asymmetric Joint Volume Supervision, which assigns different supervision preferences to the two prompt levels and employs kernelized volume ranking to enhance the consistency of cross-granularity visual–textual configurations. During inference, global and patch features are matched with both prompt levels, while anomaly responses from geometric and color-rendered views are jointly fused to produce sample-level scores and 3D point-level anomaly maps. Experiments on the MVTec3D-AD and Eyecandies datasets demonstrate that GHP-CLIP achieves competitive performance in both anomaly recognition and localization, while exhibiting effective generalization across datasets.

## Prepare Dataset

Download the original datasets: [MVTec3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad), [Eyecandies](https://eyecan-ai.github.io/eyecandies/).

The rendering images follow [PointAD](https://github.com/zqhang/PointAD). You can use the rendering data provided by PointAD, or generate them with PointAD's code.

Generate the color rendering images through `./data_preprocess`.

## Generate the dataset JSON

Generate dataset json for training:

```bash
cd generate_dataset_json
bash generate_training_datasets_class_specific.sh
bash generate_training_datasets_eye_class_specific.sh
```

Generate dataset json for testing:

```bash
bash generate_training_datasets_whole.sh
bash generate_training_datasets_eye_whole.sh
```

Change `root=` at the end of each python script to your dataset path.

## Download Pretrained Weight

Download the CLIP weights pretrained by OpenAI [[ViT-L-14-336px.pt](https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt)].

Download the PointNet++ initial weights [[best_model.pth](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/log/sem_seg/pointnet2_sem_seg/checkpoints/best_model.pth)]

Put them to `./pretrained_weights/`.

## Create Environments

```bash
conda create -n ghpclip python=3.9
conda activate ghpclip
pip install -r requirements.txt
```

## Train and Test

Training and test are included in `train.sh`. Change the data path and run this script:

```bash
bash train.sh
```

For cross-dataset test (train on MVTec3D-AD, test on Eyecandies):

```bash
bash test_cross_dataset.sh
```

