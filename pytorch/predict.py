# -*- coding: utf-8 -*-
"""
Predict floorplan representation from external images.
Usage:
    python predict.py --image path/to/your/image.jpg
    python predict.py --image path/to/image.jpg --output output_folder/
    python predict.py --image_folder path/to/images/ --output output_folder/
    python predict.py --image path/to/image.jpg --no_solver  # Skip MILP reconstruction
"""

import torch
import numpy as np
import cv2
import os
import argparse
import glob
from models.model import Model
from utils import *
from IP import reconstructFloorplan


def get_device():
    """Determine device to run model (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_image(image_path, width=256, height=256):
    """
    Load and preprocess image for model input.
    Preprocessing logic matches floorplan_dataset.py for consistency.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_image = image.copy()
    original_size = (image.shape[1], image.shape[0])  # (width, height)
    
    # Resize to model input size
    image = cv2.resize(image, (width, height))
    
    # Normalize same as dataset.py: (image / 255 - 0.5) and transpose to (C, H, W)
    image = (image.astype(np.float32) / 255 - 0.5).transpose((2, 0, 1))
    
    return image, original_image, original_size


def get_segmentation_argmax(tensor):
    """
    Compute argmax safely based on tensor shape.
    Model output can be (B, H, W, C) or (B, C, H, W).
    """
    if tensor.ndim == 4:
        # Check shape to determine dim for argmax
        # This model outputs (B, H, W, C) after transpose in forward()
        if tensor.shape[-1] < tensor.shape[1] and tensor.shape[-1] < tensor.shape[2]:
            # Shape is (B, H, W, C) -> argmax on dim=-1
            return tensor.argmax(dim=-1)
        else:
            # Shape is (B, C, H, W) -> argmax on dim=1
            return tensor.argmax(dim=1)
    elif tensor.ndim == 3:
        # (H, W, C) hoặc (C, H, W)
        if tensor.shape[-1] < tensor.shape[0] and tensor.shape[-1] < tensor.shape[1]:
            return tensor.argmax(dim=-1)
        else:
            return tensor.argmax(dim=0)
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")


def predict_single_image(model, image_path, output_dir, options, device, 
                         save_visualization=True, run_solver=True):
    """Run prediction on a single image."""
    print(f"Processing: {image_path}")
    
    # Create subdirectory for each image for clean output
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Preprocess image
    image, original_image, original_size = preprocess_image(
        image_path, options.width, options.height
    )
    
    # Add batch dimension and move to device
    image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        corner_pred, icon_pred, room_pred = model(image_tensor)
    
    # Get predictions and move to CPU/numpy
    corner_heatmaps = corner_pred[0].cpu().numpy()
    icon_heatmaps = torch.nn.functional.softmax(icon_pred[0], dim=-1).cpu().numpy()
    room_heatmaps = torch.nn.functional.softmax(room_pred[0], dim=-1).cpu().numpy()
    
    # Compute segmentation results with safe argmax
    corner_seg = get_segmentation_argmax(corner_pred)[0].cpu().numpy()
    icon_seg = get_segmentation_argmax(icon_pred)[0].cpu().numpy()
    room_seg = get_segmentation_argmax(room_pred)[0].cpu().numpy()
    
    if save_visualization:
        # Save input image (resized)
        input_vis = ((image.transpose((1, 2, 0)) + 0.5) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(image_output_dir, "input.png"), input_vis)
        
        # Save corner segmentation
        corner_vis = drawSegmentationImage(corner_seg, blackIndex=0, blackThreshold=0.5)
        cv2.imwrite(os.path.join(image_output_dir, "corners.png"), corner_vis)
        
        # Save icon segmentation
        icon_vis = drawSegmentationImage(icon_seg, blackIndex=0)
        cv2.imwrite(os.path.join(image_output_dir, "icons.png"), icon_vis)
        
        # Save room segmentation
        room_vis = drawSegmentationImage(room_seg, blackIndex=0)
        cv2.imwrite(os.path.join(image_output_dir, "rooms.png"), room_vis)
        
        print(f"  -> Segmentation saved to {image_output_dir}/")
    
    # Reconstruct floorplan using Integer Programming (if requested)
    if run_solver:
        try:
            reconstructFloorplan(
                corner_heatmaps[:, :, :NUM_WALL_CORNERS],
                corner_heatmaps[:, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS + 4],
                corner_heatmaps[:, :, -4:],
                icon_heatmaps,
                room_heatmaps,
                output_prefix=os.path.join(image_output_dir, "result_"),
                densityImage=None,
                gt_dict=None,
                gt=False,
                gap=-1,
                distanceThreshold=-1,
                lengthThreshold=-1,
                debug_prefix='predict',
                heatmapValueThresholdWall=None,
                heatmapValueThresholdDoor=None,
                heatmapValueThresholdIcon=None,
                enableAugmentation=True
            )
            print(f"  -> Reconstruction completed")
        except Exception as e:
            # Catch reconstruction error, print warning and continue
            print(f"  -> WARNING: Reconstruction failed: {e}")
            print(f"  -> Segmentation results still saved")
    else:
        print(f"  -> Skipped MILP reconstruction (--no_solver)")
    
    return {
        'corner_heatmaps': corner_heatmaps,
        'icon_heatmaps': icon_heatmaps,
        'room_heatmaps': room_heatmaps,
        'corner_seg': corner_seg,
        'icon_seg': icon_seg,
        'room_seg': room_seg,
        'output_dir': image_output_dir
    }

def main():
    parser = argparse.ArgumentParser(description='Predict floorplan from external images')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to a single image')
    parser.add_argument('--image_folder', type=str, default=None,
                        help='Path to folder containing images')
    parser.add_argument('--output', type=str, default='predictions/',
                        help='Output directory for predictions')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/floorplan/checkpoint.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--width', type=int, default=256,
                        help='Input image width')
    parser.add_argument('--height', type=int, default=256,
                        help='Input image height')
    parser.add_argument('--no_visualization', action='store_true',
                        help='Skip saving visualization images')
    parser.add_argument('--no_solver', action='store_true',
                        help='Skip MILP reconstruction (only save segmentation)')
    
    args = parser.parse_args()
    
    # Validate input
    if args.image is None and args.image_folder is None:
        parser.error("Please specify --image or --image_folder")
    
    # Determine device (GPU/CPU)
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create options object for model
    class Options:
        pass
    options = Options()
    options.width = args.width
    options.height = args.height
    options.pretrained = 0  # Don't need pretrained when loading checkpoint
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = Model(options)
    model.to(device)  # Move model to device
    model.eval()
    
    # Load checkpoint safely
    if os.path.exists(args.checkpoint):
        try:
            # Don't use weights_only=True for compatibility with old checkpoints
            state_dict = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(state_dict)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"WARNING: Failed to load checkpoint: {e}")
            print("Using randomly initialized model (results will be meaningless)")
    else:
        print(f"WARNING: Checkpoint not found at {args.checkpoint}")
        print("Using randomly initialized model (results will be meaningless)")
    
    # Collect list of images to process
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_folder:
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(glob.glob(os.path.join(args.image_folder, ext)))
            image_paths.extend(glob.glob(os.path.join(args.image_folder, ext.upper())))
    
    # Remove duplicates if any
    image_paths = list(set(image_paths))
    
    if len(image_paths) == 0:
        print("No images found to process!")
        return
    
    print(f"\nProcessing {len(image_paths)} image(s)...\n")
    
    # Process each image
    successful = 0
    failed = 0
    for image_path in image_paths:
        try:
            predict_single_image(
                model, 
                image_path, 
                args.output, 
                options,
                device=device,
                save_visualization=not args.no_visualization,
                run_solver=not args.no_solver
            )
            successful += 1
        except Exception as e:
            print(f"ERROR processing {image_path}: {e}")
            failed += 1
            # Continue processing other images, don't crash mid-batch
            continue
    
    print(f"\nDone! Processed {successful}/{len(image_paths)} images successfully.")
    if failed > 0:
        print(f"Failed: {failed} image(s)")
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
