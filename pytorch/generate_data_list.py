"""
Script to generate train.txt and test.txt from the vector_graphics_floorplans folder
"""
import os
import random

def find_all_samples(base_path):
    """Find all image files and their corresponding annotation files"""
    samples = []
    image_folder = os.path.join(base_path, 'floorplan_image')
    annotation_folder = os.path.join(base_path, '..', 'data', 'floorplan_representation')
    
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                # Get relative path from image folder
                rel_path = os.path.relpath(root, image_folder)
                image_rel_path = os.path.join(rel_path, file)
                
                # Corresponding annotation path
                annotation_name = os.path.splitext(file)[0] + '.txt'
                annotation_rel_path = os.path.join(rel_path, annotation_name)
                
                # Full paths for checking
                full_annotation_path = os.path.join(annotation_folder, annotation_rel_path)
                
                if os.path.exists(full_annotation_path):
                    # Convert to forward slashes and proper format
                    image_path = '../vector_graphics_floorplans/floorplan_image/' + image_rel_path.replace('\\', '/')
                    annotation_path = 'floorplan_representation/' + annotation_rel_path.replace('\\', '/')
                    samples.append((image_path, annotation_path))
                else:
                    print(f"Warning: No annotation for {image_rel_path}")
    
    return samples

def main():
    base_path = '../vector_graphics_floorplans'
    output_folder = '../data'
    
    print("Scanning for samples...")
    samples = find_all_samples(base_path)
    print(f"Found {len(samples)} samples with both image and annotation")
    
    if len(samples) == 0:
        print("No samples found!")
        return
    
    # Shuffle and split: 90% train, 10% test
    random.seed(42)
    random.shuffle(samples)
    
    split_idx = int(len(samples) * 0.9)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]
    
    # Ensure at least 1 test sample
    if len(test_samples) == 0 and len(train_samples) > 0:
        test_samples = [train_samples.pop()]
    
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")
    
    # Write train.txt
    with open(os.path.join(output_folder, 'train.txt'), 'w') as f:
        for image_path, annotation_path in train_samples:
            f.write(f"{image_path}\t{annotation_path}\n")
    print(f"Written {output_folder}/train.txt")
    
    # Write test.txt
    with open(os.path.join(output_folder, 'test.txt'), 'w') as f:
        for image_path, annotation_path in test_samples:
            f.write(f"{image_path}\t{annotation_path}\n")
    print(f"Written {output_folder}/test.txt")
    
    # Show sample
    print("\nSample entries:")
    if train_samples:
        print(f"  Train: {train_samples[0]}")
    if test_samples:
        print(f"  Test: {test_samples[0]}")

if __name__ == '__main__':
    main()
