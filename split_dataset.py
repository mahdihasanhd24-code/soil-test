import os
import shutil
import random
import sys

def split_data(source_dir, dest_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits images from source_dir into dest_dir/train, dest_dir/val, dest_dir/test
    with the given ratios. Shuffles with a fixed seed.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-9, "Ratios must sum to 1.0"
    
    random.seed(seed)
    
    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")
        
    # Get all class directories
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    if not classes:
        raise ValueError(f"No folders found in source directory {source_dir}.")
        
    print(f"Found {len(classes)} classes to split: {classes}")
    
    # Define split subdirs
    splits = ['train', 'val', 'test']
    
    # If destination exists, clear it to avoid accumulating or mixing old splits
    if os.path.exists(dest_dir):
        print(f"Clearing existing split directory: {dest_dir}")
        shutil.rmtree(dest_dir)
        
    # Create directories
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)
            
    summary_counts = {}
    
    for cls in classes:
        cls_dir = os.path.join(source_dir, cls)
        # Find all valid image files
        files = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f)) and f.lower().endswith(valid_extensions)]
        
        # Shuffle files
        random.shuffle(files)
        
        total_files = len(files)
        if total_files == 0:
            print(f"Warning: No images found in {cls_dir}")
            continue
            
        # Calculate split indices
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)
        
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]
        
        # Handle cases where rounding leaves test set empty
        if len(test_files) == 0 and len(files) >= 3:
            # Shift one from val to test
            test_files = [val_files.pop()]
            
        print(f"Class '{cls}': {total_files} total -> Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        summary_counts[cls] = {
            'total': total_files,
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files)
        }
        
        # Copy files
        for f in train_files:
            shutil.copy2(os.path.join(cls_dir, f), os.path.join(dest_dir, 'train', cls, f))
        for f in val_files:
            shutil.copy2(os.path.join(cls_dir, f), os.path.join(dest_dir, 'val', cls, f))
        for f in test_files:
            shutil.copy2(os.path.join(cls_dir, f), os.path.join(dest_dir, 'test', cls, f))
            
    print("Dataset split successfully completed!")
    return summary_counts

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(base_dir, "DataSet")
    dst = os.path.join(base_dir, "dataset_split")
    try:
        split_data(src, dst)
    except Exception as e:
        print(f"Error splitting dataset: {e}")
        sys.exit(1)
