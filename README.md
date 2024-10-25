# Gated-MCBAM: A Cross-Modal Attention Framework with Dual-Stream Architecture for Multi-Source Remote Sensing Segmentation

This repository contains the implementation of Gated-MCBAM, an innovative dual-stream framework that combines cross-modal attention and gating mechanisms for multi-source remote sensing segmentation. Our approach effectively integrates SAR and optical remote sensing data through a sophisticated attention mechanism.

## Model Architecture

Our model features:
- Dual-stream architecture for processing SAR and MSI data
- Cross-modal attention mechanism for feature interaction
- Gating mechanism for adaptive feature selection
- Multi-scale feature fusion
- Integration of Swin Transformer and ResNet backbones

## Preprocessing

### 1. 12-Channel to 10-Channel Conversion
- We use both 12-channel and 10-channel data for ensemble predictions
- The original 12-channel data from YREB-dataset is converted to 10-channel format
- Conversion script: `tools/dataset_converters/12ch-10ch.py`
- Output is stored in the `multisen` folder

### 2. SAR Data Processing
- Process VV and VH channels into 3-channel format
- Create a new directory called 'SAR_AVG_TIF' containing:
  - Channel 1: VV
  - Channel 2: VH
  - Channel 3: (VV+VH)/2
- Processing script: `tools/dataset_converters/new_channel_yreb.py`

## Model Weights

Pre-trained model weights can be downloaded from:
[Google Drive Link](https://drive.google.com/file/d/1fKRVMwmWSFI2TxDi-9z8e1bGPigLlm-7/view?usp=drive_link)

## Testing

### Individual Model Testing
You can test individual models using:
```bash
python tools/test.py \
    --config path/to/config.py \
    --checkpoint path/to/weights.pth
```

### Ensemble Testing
1. Download the model weights from our Google Drive
2. Save the weights in your local directory
3. Navigate to the `workdir` folder
4. Run `ensemble.py` with appropriate config files and weight paths


## Directory Structure
```
├── tools
│   ├── test.py
│   └── train.py
│   └── dataset_converters
│       └── new_channel_yreb.py
        └── 12ch-10ch.py
└── workdir
    └── ensemble.py
```

## Training

Training documentation will be updated soon.

## Contact

For any questions or issues, please contact:
- Email: jeongho.min@unist.ac.kr

## License

[License information to be added]

## Citation

If you find this work useful in your research, please consider citing:
```
[Citation information to be added]
```
