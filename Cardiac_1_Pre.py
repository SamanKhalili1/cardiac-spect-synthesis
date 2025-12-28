import os
import shutil
import logging
import numpy as np
import pydicom
from sklearn.model_selection import KFold
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('half_projection_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== UTILITY FUNCTIONS ==========

def validate_and_fix_uid(uid_value):
    """Enhanced UID validation with better error handling"""
    try:
        if not uid_value or str(uid_value).strip() == "":
            return pydicom.uid.generate_uid()
        
        uid_clean = str(uid_value).strip().strip('.')
        
        if not uid_clean or not all(c.isdigit() or c == '.' for c in uid_clean):
            return pydicom.uid.generate_uid()
        
        if '..' in uid_clean or len(uid_clean) > 64:
            return pydicom.uid.generate_uid()
        
        components = uid_clean.split('.')
        for component in components:
            if not component or not component.isdigit():
                return pydicom.uid.generate_uid()
        
        return uid_clean
        
    except Exception:
        return pydicom.uid.generate_uid()

def sanitize_dicom_metadata(dicom_obj):
    """Enhanced DICOM metadata sanitization with compression handling"""
    try:
        # Fix UIDs
        dicom_obj.SOPInstanceUID = validate_and_fix_uid(getattr(dicom_obj, 'SOPInstanceUID', None))
        
        if hasattr(dicom_obj, 'StudyInstanceUID'):
            dicom_obj.StudyInstanceUID = validate_and_fix_uid(dicom_obj.StudyInstanceUID)
        
        if hasattr(dicom_obj, 'SeriesInstanceUID'):
            dicom_obj.SeriesInstanceUID = validate_and_fix_uid(dicom_obj.SeriesInstanceUID)
        
        # Sync file meta
        if hasattr(dicom_obj, 'file_meta') and dicom_obj.file_meta:
            if hasattr(dicom_obj.file_meta, 'MediaStorageSOPInstanceUID'):
                dicom_obj.file_meta.MediaStorageSOPInstanceUID = dicom_obj.SOPInstanceUID
            
            # Ensure uncompressed transfer syntax
            dicom_obj.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        # Set required tags for uncompressed DICOM
        required_tags = {
            'SoftwareVersions': 'Half_Projection_Processor_v1.0',
            'Modality': 'OT',
            'PhotometricInterpretation': 'MONOCHROME2',
            'SamplesPerPixel': 1,
            'BitsAllocated': 16,
            'BitsStored': 16,
            'HighBit': 15,
            'PixelRepresentation': 0
        }
        
        for tag_name, default_value in required_tags.items():
            setattr(dicom_obj, tag_name, default_value)
        
    except Exception as e:
        logger.error(f"DICOM sanitization failed: {e}")

# ========== DATA LOADING ==========

def load_dicom_file(file_path):
    """Load and validate DICOM file with compression handling"""
    try:
        # Load with decompression if needed
        dicom_data = pydicom.dcmread(file_path, force=True)
        
        # Handle compressed data
        if hasattr(dicom_data, 'file_meta') and dicom_data.file_meta:
            if dicom_data.file_meta.TransferSyntaxUID.is_compressed:
                dicom_data.decompress()
        
        sanitize_dicom_metadata(dicom_data)
        
        if not hasattr(dicom_data, 'pixel_array'):
            raise ValueError("No pixel data found")
        
        pixel_array = dicom_data.pixel_array
        if pixel_array is None or pixel_array.size == 0:
            raise ValueError("Empty pixel array")
        
        # Force decompression and ensure proper data type
        pixel_array = np.array(pixel_array, copy=True)
        
        # Convert to uint16 for consistency
        if pixel_array.dtype != np.uint16:
            if pixel_array.max() <= 1.0:
                pixel_array = (pixel_array * 65535).astype(np.uint16)
            elif pixel_array.dtype == np.int16:
                pixel_array = pixel_array.astype(np.int32) + 32768
                pixel_array = np.clip(pixel_array, 0, 65535).astype(np.uint16)
            else:
                pixel_array = np.clip(pixel_array, 0, 65535).astype(np.uint16)
        
        return pixel_array, dicom_data, True
        
    except Exception as e:
        logger.error(f"Failed to load DICOM file {file_path}: {e}")
        return None, None, False

# ========== GLOBAL NORMALIZATION STATISTICS ==========

def compute_global_normalization_stats(input_dir, sample_size=None, force_recompute=False):
    """
    CRITICAL: Compute global normalization statistics from training data
    
    This function computes statistics that will be used for consistent
    normalization across preprocessing and training.
    
    Args:
        input_dir: Directory containing DICOM files
        sample_size: Number of files to sample (None = use all)
        force_recompute: Force recomputation even if stats file exists
    
    Returns:
        dict: {'p1': percentile_1, 'p99': percentile_99, 'mean': mean, 'std': std, ...}
    """
    logger.info("="*80)
    logger.info("COMPUTING GLOBAL NORMALIZATION STATISTICS")
    logger.info("="*80)
    
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return None
    
    # Get all DICOM files
    dicom_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.dcm')])
    
    if len(dicom_files) == 0:
        logger.error(f"No DICOM files found in {input_dir}")
        return None
    
    # Sample if needed
    if sample_size and len(dicom_files) > sample_size:
        np.random.seed(42)
        dicom_files = np.random.choice(dicom_files, sample_size, replace=False).tolist()
        logger.info(f"Sampling {sample_size} files from {len(dicom_files)} total files")
    else:
        logger.info(f"Using all {len(dicom_files)} files for statistics")
    
    all_values = []
    successful_loads = 0
    
    for i, filename in enumerate(dicom_files):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info(f"  Processing file {i+1}/{len(dicom_files)}...")
        
        try:
            file_path = os.path.join(input_dir, filename)
            pixel_array, _, success = load_dicom_file(file_path)
            
            if success and pixel_array is not None:
                # Only collect non-zero values to avoid bias from padding
                non_zero_values = pixel_array[pixel_array > 0]
                if len(non_zero_values) > 0:
                    # Sample to avoid memory issues
                    if len(non_zero_values) > 10000:
                        sample_indices = np.random.choice(len(non_zero_values), 10000, replace=False)
                        all_values.extend(non_zero_values[sample_indices].flatten())
                    else:
                        all_values.extend(non_zero_values.flatten())
                    successful_loads += 1
        except Exception as e:
            logger.debug(f"Skipping {filename} in stats computation: {e}")
            continue
    
    if len(all_values) == 0:
        logger.error("No valid data found for global statistics!")
        return None
    
    all_values = np.array(all_values, dtype=np.float64)
    
    stats = {
        'p1': float(np.percentile(all_values, 1)),
        'p99': float(np.percentile(all_values, 99)),
        'p5': float(np.percentile(all_values, 5)),
        'p95': float(np.percentile(all_values, 95)),
        'mean': float(np.mean(all_values)),
        'std': float(np.std(all_values)),
        'median': float(np.median(all_values)),
        'min': float(np.min(all_values)),
        'max': float(np.max(all_values)),
        'files_sampled': len(dicom_files),
        'files_successful': successful_loads,
        'total_voxels': len(all_values),
        'computation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    logger.info(f"Global statistics computed from {successful_loads}/{len(dicom_files)} files:")
    logger.info(f"  p1={stats['p1']:.2f}, p99={stats['p99']:.2f}")
    logger.info(f"  mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    logger.info(f"  median={stats['median']:.2f}")
    logger.info(f"  range=[{stats['min']:.2f}, {stats['max']:.2f}]")
    logger.info(f"  Total voxels analyzed: {stats['total_voxels']:,}")
    logger.info("="*80)
    
    return stats

# ========== HALF PROJECTION CREATION ==========

def create_alternating_half_projection(volume_data):
    """
    Create half projection by zeroing odd frames (1, 3, 5, ..., 31)
    Keep even frames (0, 2, 4, ..., 30)
    
    Args:
        volume_data: Input volume data (frames, height, width)
    
    Returns:
        half_volume: Half projection with odd frames zeroed
    """
    # Handle 2D case
    if volume_data.ndim == 2:
        volume_data = np.expand_dims(volume_data, axis=0)
    
    half_volume = volume_data.copy()
    
    # Zero odd frames (1, 3, 5, ..., 31)
    num_frames = half_volume.shape[0]
    odd_frames = list(range(1, num_frames, 2))
    
    for frame_idx in odd_frames:
        if frame_idx < half_volume.shape[0]:
            half_volume[frame_idx] = np.zeros_like(half_volume[frame_idx])
    
    # Log statistics
    even_frames = list(range(0, num_frames, 2))
    original_nonzero = np.sum(volume_data > 0)
    half_nonzero = np.sum(half_volume > 0)
    
    logger.debug(f"Half projection (alternating) created:")
    logger.debug(f"  Total frames: {num_frames}")
    logger.debug(f"  Frames kept (even): {len(even_frames)}")
    logger.debug(f"  Frames zeroed (odd): {len(odd_frames)}")
    logger.debug(f"  Data retention: {half_nonzero/original_nonzero*100:.1f}%")
    
    return half_volume

# ========== DICOM SAVING ==========

def save_dicom_volume(volume_data, original_dicom, output_path, description="Processed"):
    """Save volume data as DICOM file"""
    try:
        new_dicom = original_dicom.copy()
        
        # Force to uncompressed transfer syntax
        new_dicom.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        # High precision uint16 conversion
        if volume_data.dtype != np.uint16:
            if volume_data.max() <= 1.0:
                volume_data = np.round(volume_data * 65535.0).astype(np.uint16)
            else:
                volume_data = np.clip(volume_data, 0, 65535).astype(np.uint16)
        
        # Handle multi-frame
        if volume_data.ndim == 3 and volume_data.shape[0] > 1:
            new_dicom.NumberOfFrames = volume_data.shape[0]
            new_dicom.Rows = volume_data.shape[1]
            new_dicom.Columns = volume_data.shape[2]
            
            volume_data = np.ascontiguousarray(volume_data, dtype=np.uint16)
            pixel_data = volume_data.tobytes()
            
        else:
            if volume_data.ndim == 3:
                volume_data = volume_data[volume_data.shape[0] // 2]
            
            new_dicom.Rows = volume_data.shape[0]
            new_dicom.Columns = volume_data.shape[1]
            
            volume_data = np.ascontiguousarray(volume_data, dtype=np.uint16)
            pixel_data = volume_data.tobytes()
            
            if hasattr(new_dicom, 'NumberOfFrames'):
                delattr(new_dicom, 'NumberOfFrames')
        
        # Set pixel data and metadata
        new_dicom.PixelData = pixel_data
        new_dicom.SeriesDescription = description
        new_dicom.SOPInstanceUID = pydicom.uid.generate_uid()
        
        # Ensure proper DICOM tags
        new_dicom.BitsAllocated = 16
        new_dicom.BitsStored = 16
        new_dicom.HighBit = 15
        new_dicom.PixelRepresentation = 0
        new_dicom.PhotometricInterpretation = 'MONOCHROME2'
        new_dicom.SamplesPerPixel = 1
        
        # Update timestamps
        now = datetime.now()
        new_dicom.StudyDate = now.strftime('%Y%m%d')
        new_dicom.StudyTime = now.strftime('%H%M%S')
        new_dicom.SeriesDate = now.strftime('%Y%m%d')
        new_dicom.SeriesTime = now.strftime('%H%M%S')
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        new_dicom.save_as(output_path, enforce_file_format=True)
        return True
        
    except Exception as e:
        logger.error(f"Failed to save DICOM {output_path}: {e}")
        return False

# ========== MAIN PROCESSING ==========

def process_half_projections(half_input_dir, full_input_dir, output_base_dir):
    """
    Process Half_Time files by zeroing odd frames
    
    Args:
        half_input_dir: Directory containing Half_Time DICOM files (L64_Dicom)
        full_input_dir: Directory containing Full projection files (H64_Dicom)
        output_base_dir: Base output directory (1_Pre)
    
    Returns:
        dict: Processing statistics
    """
    logger.info("="*80)
    logger.info("STEP 1: Processing Half Projections")
    logger.info("="*80)
    
    # Create output directory
    output_dir = os.path.join(output_base_dir, "1_Pre_alternating")
    half_output = os.path.join(output_dir, "Half_Projections")
    full_output = os.path.join(output_dir, "Full_Projections")
    
    os.makedirs(half_output, exist_ok=True)
    os.makedirs(full_output, exist_ok=True)
    
    # Get all DICOM files
    half_files = sorted([f for f in os.listdir(half_input_dir) if f.endswith('.dcm')])
    full_files = sorted([f for f in os.listdir(full_input_dir) if f.endswith('.dcm')])
    
    # Find common files
    common_files = sorted(list(set(half_files) & set(full_files)))
    
    logger.info(f"Found {len(half_files)} Half_Time files")
    logger.info(f"Found {len(full_files)} Full projection files")
    logger.info(f"Common files: {len(common_files)}")
    
    if len(common_files) == 0:
        logger.error("No common files found between Half and Full directories!")
        return None
    
    stats = {
        'total_files': len(common_files),
        'processed_successfully': 0,
        'failed_files': 0,
        'file_list': []
    }
    
    # Process each file
    for idx, filename in enumerate(common_files):
        try:
            # Load Half_Time file
            half_path = os.path.join(half_input_dir, filename)
            half_array, half_dicom, half_success = load_dicom_file(half_path)
            
            if not half_success or half_array is None:
                logger.warning(f"Skipping {filename} - Half file load failed")
                stats['failed_files'] += 1
                continue
            
            # Load Full file
            full_path = os.path.join(full_input_dir, filename)
            full_array, full_dicom, full_success = load_dicom_file(full_path)
            
            if not full_success or full_array is None:
                logger.warning(f"Skipping {filename} - Full file load failed")
                stats['failed_files'] += 1
                continue
            
            # Create alternating half projection by zeroing odd frames
            half_processed = create_alternating_half_projection(half_array)
            
            # Save processed Half projection
            half_out_path = os.path.join(half_output, filename)
            half_saved = save_dicom_volume(
                half_processed, 
                half_dicom, 
                half_out_path, 
                "Alternating_Half_Projection"
            )
            
            # Copy Full projection (Ground Truth)
            full_out_path = os.path.join(full_output, filename)
            full_saved = save_dicom_volume(
                full_array, 
                full_dicom, 
                full_out_path, 
                "Full_Projection_Ground_Truth"
            )
            
            if half_saved and full_saved:
                stats['processed_successfully'] += 1
                stats['file_list'].append(filename)
                
                if (idx + 1) % 50 == 0 or idx == 0:
                    logger.info(f"Processed {idx + 1}/{len(common_files)} files")
            else:
                stats['failed_files'] += 1
                
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            stats['failed_files'] += 1
            continue
    
    logger.info("="*80)
    logger.info(f"Processing completed: {stats['processed_successfully']}/{stats['total_files']} files")
    logger.info("="*80)
    
    return stats

# ========== K-FOLD DATASET CREATION ==========

def create_kfold_datasets(output_base_dir, stats, n_splits=5, 
                         train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                         random_state=42):
    """
    Create K-Fold cross-validation datasets with 70/15/15 split
    
    Args:
        output_base_dir: Base output directory
        stats: Processing statistics
        n_splits: Number of folds
        train_ratio: Training set ratio (0.70)
        val_ratio: Validation set ratio (0.15)
        test_ratio: Test set ratio (0.15)
        random_state: Random seed
    
    Returns:
        dict: Fold information
    """
    logger.info("="*80)
    logger.info(f"STEP 2: Creating {n_splits}-Fold Cross-Validation Datasets")
    logger.info(f"Split ratios - Train: {train_ratio*100:.0f}%, Val: {val_ratio*100:.0f}%, Test: {test_ratio*100:.0f}%")
    logger.info("="*80)
    
    output_dir = os.path.join(output_base_dir, "1_Pre_alternating")
    half_dir = os.path.join(output_dir, "Half_Projections")
    full_dir = os.path.join(output_dir, "Full_Projections")
    
    all_files = sorted(stats['file_list'])
    
    if len(all_files) == 0:
        logger.error("No files available for K-Fold splitting")
        return None
    
    logger.info(f"Total files for K-Fold: {len(all_files)}")
    
    # Create KFold splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_info = {}
    
    # Create folds
    for fold_idx, (train_val_indices, test_indices) in enumerate(kf.split(all_files), 1):
        logger.info(f"\nCreating Fold {fold_idx}...")
        
        # Get test files (15%)
        test_files = [all_files[i] for i in test_indices]
        
        # Remaining files for train + val (85%)
        train_val_files = [all_files[i] for i in train_val_indices]
        
        # Split train_val into train (70% of total) and val (15% of total)
        # This means train is 70/85 of train_val, val is 15/85 of train_val
        train_split_point = int(len(train_val_files) * (train_ratio / (train_ratio + val_ratio)))
        
        train_files = train_val_files[:train_split_point]
        val_files = train_val_files[train_split_point:]
        
        logger.info(f"  Train: {len(train_files)} files ({len(train_files)/len(all_files)*100:.1f}%)")
        logger.info(f"  Val:   {len(val_files)} files ({len(val_files)/len(all_files)*100:.1f}%)")
        logger.info(f"  Test:  {len(test_files)} files ({len(test_files)/len(all_files)*100:.1f}%)")
        
        # Create fold directories
        fold_dir = os.path.join(output_dir, f"Fold_{fold_idx}")
        
        for split in ['Train', 'Val', 'Test']:
            for proj_type in ['Full', 'Half']:
                split_dir = os.path.join(fold_dir, f"{split}_{proj_type}")
                os.makedirs(split_dir, exist_ok=True)
        
        # Copy files
        def copy_files_to_split(file_list, split_name):
            for f in file_list:
                full_src = os.path.join(full_dir, f)
                half_src = os.path.join(half_dir, f)
                
                full_dst = os.path.join(fold_dir, f"{split_name}_Full", f)
                half_dst = os.path.join(fold_dir, f"{split_name}_Half", f)
                
                if os.path.exists(full_src):
                    shutil.copy2(full_src, full_dst)
                if os.path.exists(half_src):
                    shutil.copy2(half_src, half_dst)
        
        copy_files_to_split(train_files, "Train")
        copy_files_to_split(val_files, "Val")
        copy_files_to_split(test_files, "Test")
        
        # Store fold info
        fold_info[f"fold_{fold_idx}"] = {
            'train_files': train_files,
            'val_files': val_files,
            'test_files': test_files,
            'n_train': len(train_files),
            'n_val': len(val_files),
            'n_test': len(test_files),
            'train_ratio': len(train_files) / len(all_files),
            'val_ratio': len(val_files) / len(all_files),
            'test_ratio': len(test_files) / len(all_files)
        }
    
    # Save fold information
    fold_info_file = os.path.join(output_dir, "kfold_info.json")
    with open(fold_info_file, 'w') as f:
        json.dump(fold_info, f, indent=2)
    
    logger.info(f"\nK-Fold information saved to: {fold_info_file}")
    logger.info("="*80)
    
    return fold_info

# ========== MAIN EXECUTION ==========

def main():
    """
    Main execution function with global stats computation
    """
    print("="*80)
    print("HALF PROJECTION PROCESSOR WITH K-FOLD CROSS-VALIDATION")
    print("AND GLOBAL NORMALIZATION STATISTICS")
    print("="*80)
    
    # Configuration
    half_input_dir = r"E:\Project\Cardiac\L64_Dicom"
    full_input_dir = r"E:\Project\Cardiac\H64_Dicom"
    output_base_dir = r"E:\Project\Cardiac"
    
    # Verify input directories exist
    if not os.path.exists(half_input_dir):
        logger.error(f"Half input directory not found: {half_input_dir}")
        return None, None, None
    
    if not os.path.exists(full_input_dir):
        logger.error(f"Full input directory not found: {full_input_dir}")
        return None, None, None
    
    logger.info(f"Half input directory: {half_input_dir}")
    logger.info(f"Full input directory: {full_input_dir}")
    logger.info(f"Output base directory: {output_base_dir}")
    
    # STEP 0: Compute global normalization statistics
    logger.info("\n" + "="*80)
    logger.info("STEP 0: Computing Global Normalization Statistics")
    logger.info("="*80)
    
    global_stats = compute_global_normalization_stats(
        input_dir=full_input_dir,  # Use Full projections for stats
        sample_size=100,  # Sample 100 files for efficiency
        force_recompute=False
    )
    
    if global_stats is None:
        logger.error("Failed to compute global statistics - aborting")
        return None, None, None
    
    # Save global stats to base directory for training pipeline
    stats_file = os.path.join(output_base_dir, "global_normalization_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(global_stats, f, indent=2)
    logger.info(f"Global statistics saved to: {stats_file}")
    
    # Step 1: Process Half projections
    stats = process_half_projections(
        half_input_dir=half_input_dir,
        full_input_dir=full_input_dir,
        output_base_dir=output_base_dir
    )
    
    if stats is None or stats['processed_successfully'] == 0:
        logger.error("Processing failed - no files processed successfully")
        return None, None, None
    
    # Step 2: Create K-Fold datasets (70/15/15 split)
    fold_info = create_kfold_datasets(
        output_base_dir=output_base_dir,
        stats=stats,
        n_splits=5,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )
    
    # Final summary
    print("="*80)
    print("PROCESSING COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Total files processed: {stats['processed_successfully']}/{stats['total_files']}")
    print(f"K-Fold splits created: {len(fold_info)} folds")
    print(f"\nGlobal Statistics:")
    print(f"  p1:   {global_stats['p1']:.2f}")
    print(f"  p99:  {global_stats['p99']:.2f}")
    print(f"  mean: {global_stats['mean']:.2f}")
    print(f"  std:  {global_stats['std']:.2f}")
    print(f"\nOutput structure:")
    print(f"  {output_base_dir}\\")
    print(f"    ├── global_normalization_stats.json")
    print(f"    └── 1_Pre_alternating\\")
    print(f"        ├── Half_Projections\\")
    print(f"        ├── Full_Projections\\")
    print(f"        ├── Fold_1\\")
    print(f"        │   ├── Train_Half\\")
    print(f"        │   ├── Train_Full\\")
    print(f"        │   ├── Val_Half\\")
    print(f"        │   ├── Val_Full\\")
    print(f"        │   ├── Test_Half\\")
    print(f"        │   └── Test_Full\\")
    print(f"        ├── Fold_2\\ ... Fold_5\\")
    print(f"        └── kfold_info.json")
    
    # Display fold statistics
    print(f"\nFold Statistics:")
    for fold_name, fold_data in fold_info.items():
        print(f"  {fold_name}:")
        print(f"    Train: {fold_data['n_train']} files ({fold_data['train_ratio']*100:.1f}%)")
        print(f"    Val:   {fold_data['n_val']} files ({fold_data['val_ratio']*100:.1f}%)")
        print(f"    Test:  {fold_data['n_test']} files ({fold_data['test_ratio']*100:.1f}%)")
    
    print("="*80)
    
    return stats, fold_info, global_stats

# ========== ENTRY POINT ==========

if __name__ == "__main__":
    stats, fold_info, global_stats = main()
    
    if stats and fold_info and global_stats:
        print("\n✓ All processing completed successfully!")
        print("\nFiles generated:")
        print("  1. E:\\Project\\Cardiac\\global_normalization_stats.json")
        print("  2. E:\\Project\\Cardiac\\1_Pre_alternating\\Half_Projections\\")
        print("  3. E:\\Project\\Cardiac\\1_Pre_alternating\\Full_Projections\\")
        print("  4. E:\\Project\\Cardiac\\1_Pre_alternating\\Fold_1\\ ... Fold_5\\")
        print("  5. E:\\Project\\Cardiac\\1_Pre_alternating\\kfold_info.json")
        print("\nNext steps:")
        print("  1. Verify the global_normalization_stats.json file")
        print("  2. Check the output directories")
        print("  3. Run the training pipeline (it will use global stats)")
    else:
        print("\n✗ Processing encountered errors - check logs")
    
    print("\nScript execution completed.")



