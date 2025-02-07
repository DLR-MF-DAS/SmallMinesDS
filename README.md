# SmallMinesDS

This repository contains the code and scripts required to reproduce the results presented in our paper *SmallMinesDS*.

## Prerequisites
- **Conda** (for managing environments)
- **NVIDIA GPU** (for efficient computation)

## Setting Up Environments
Two separate virtual environments are required:
1. **Terratorch Environment** (for TerraTorch dependencies)
2. **SAM2 Environment** (for Segment Anything Model v2 dependencies)

These environments are necessary as they rely on different PyTorch versions.

### Creating the TerraTorch Environment
```bash
conda create -n terratorch python=3.11
conda activate terratorch
pip install -r requirements.txt
conda deactivate
```

### Creating the SAM2 Environment
```bash
conda create -n sam2 python=3.11
conda activate sam2
chmod +x install_sam2.sh
bash install_sam2.sh
mv ft-sam2.py sam2/
conda deactivate
```

## Fine-tuning Prithvi-2
Fine-tune Prithvi-2 using our dataset.

### Fine-tuning with the 300M Model
```bash
conda activate terratorch
python train-prithvi-v2-300.py
```

### Fine-tuning with the 600M Model
```bash
conda activate terratorch
python train-prithvi-v2-600.py
```

## Training ResNet50 from Scratch
To compare with Prithvi-2, we train ResNet50 from scratch using six spectral bands: **Blue, Green, Red, Narrow NIR, SWIR, and SWIR 2**.

```bash
conda activate terratorch
python train-resnet50-6bands.py
```

## Fine-tuning the Segment Anything Model v2 (SAM2)
Segment Anything Model v2 is a foundational model designed for promptable visual segmentation in RGB images and videos. We fine-tune it using the RGB channels of our dataset.

```bash
conda activate sam2
python sam2/ft-sam2.py
```

## Fine-tuning ResNet50 Pretrained on ImageNet
For a fair comparison in the RGB domain, we fine-tune a ResNet50 model pretrained on ImageNet and compare its performance with SAM2.

```bash
conda activate terratorch
python ft-resnet50.py
```