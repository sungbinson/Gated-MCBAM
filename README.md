# Gated-based MultiCbam for Multi-modal Semantic Segmentation

This repository contains the implementation of our proposed Gated-based MultiCbam model for multi-modal semantic segmentation.

## Overview

Our model utilizes both 12-channel and 10-channel inputs for ensemble predictions, along with 3-channel VV/VH data for multi-modal semantic segmentation.

## Preprocessing

### 1. 12-Channel to 10-Channel Conversion
- We use both 12-channel and 10-channel data for ensemble predictions
- The original 12-channel data from YREB-dataset is converted to 10-channel format
- Conversion script: `12ch-10ch.py`
- Output is stored in the `multisen` folder

### 2. VV/VH Channel Processing
- VV and VH channels are processed into 3-channel format
- Processing script: `tools/dataset_converters/new_channel_yreb.py`

## Testing

1. Download the model weights from our [Google Drive]
2. Save the weights in your local directory
3. Navigate to the `workdir` folder
4. Run `ensemble.py` with appropriate config file and weight paths

Example:
```bash
python ensemble.py \
    --config path/to/config.py \
    --checkpoint path/to/weights.pth
```

## Training

Training documentation will be updated soon.

## Directory Structure
```
├── 12ch-10ch.py
├── tools
│   └── dataset_converters
│       └── new_channel_yreb.py
└── workdir
    └── ensemble.py
```

## Contact

For any questions or issues, please contact:
- Email: min.jeongho@lumir.space

## License

[License information to be added]

## Citation

[Citation information to be added]
