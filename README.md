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
- Conversion script: `12ch-10ch.py`
- Output is stored in the `multisen` folder

### 2. VV/VH Channel Processing
- VV and VH channels are processed into 3-channel format
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

Example:
```bash
python ensemble.py \
    --config path/to/config.py \
    --checkpoint path/to/weights.pth
```

## Directory Structure
```
├── 12ch-10ch.py
├── tools
│   ├── test.py
│   └── dataset_converters
│       └── new_channel_yreb.py
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
