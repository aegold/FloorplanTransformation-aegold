# FloorplanTransformation - PyTorch Version

This is the PyTorch implementation (updated for Python 3.10+ and PyTorch 2.0+).

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.x or 12.x (for GPU support)

## Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Training

```bash
python train.py --restore=0
```

Set `--restore=1` to resume training from a checkpoint.

### Training Options

- `--batchSize`: Batch size (default: 16)
- `--LR`: Learning rate (default: 2.5e-4)
- `--numEpochs`: Number of epochs (default: 1000)
- `--width`, `--height`: Input image size (default: 256x256)

## Pre-trained model

Download the pretrained checkpoint from [Google Drive](https://drive.google.com/open?id=1e5c7308fdoCMRv0w-XduWqyjYPV4JWHS).

Place it in `checkpoint/floorplan/checkpoint.pth`.

## Testing

```bash
python train.py --task=test
```

## GPU Requirements

- Minimum: 6GB VRAM (reduce batch size if needed)
- Recommended: 12GB+ VRAM
- Tested on: RTX 3090 (24GB)
