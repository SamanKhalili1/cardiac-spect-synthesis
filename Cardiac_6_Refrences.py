"""
Reference Image Generator - CLEAN VERSION (No titles, labels, or scales)
=========================================================================
Automatically processes COMMON files between Full and Generated directories.
Produces pure image output without any annotations.
"""

import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURATION
# =====================================================================

class Config:
    BASE_DIR = r"E:\Project\Cardiac"
    TEST_FULL_DIR = os.path.join(BASE_DIR, "4_Recon_Full")
    TEST_HALF_DIR = os.path.join(BASE_DIR, "4_Recon_Generated")
    OUTPUT_DIR = os.path.join(BASE_DIR, "6_References")
    
    # Slice selection FROM THE 32-SLICE VOLUME (indices 0 to 31)
    START_SLICE = 8
    END_SLICE = 23  # 16 slices
    
    GRID_ROWS = 4
    GRID_COLS = 4
    COLORMAP = 'viridis'
    DPI = 300
    FIGSIZE = (16, 10)
    MAX_FILES = 10  # Limit number of generated references

# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def load_dicom_volume(file_path):
    """Load DICOM volume and normalize to [0,1]"""
    try:
        dcm = pydicom.dcmread(file_path, force=True)
        volume = dcm.pixel_array.astype(np.float32)
        if volume.ndim == 2:
            volume = np.expand_dims(volume, axis=0)
        if volume.max() > 0:
            volume = volume / volume.max()
        return volume, True
    except Exception as e:
        print(f"    ERROR loading {file_path}: {e}")
        return None, False

def select_slices(volume, start_idx, end_idx):
    """Select and pad slices to exactly 16 slices"""
    if volume.shape[0] != 32:
        print(f"    WARNING: Expected 32 slices, got {volume.shape[0]}")
    
    start_idx = max(0, min(start_idx, volume.shape[0] - 1))
    end_idx = max(start_idx, min(end_idx, volume.shape[0] - 1))
    selected = volume[start_idx:end_idx+1]
    
    if selected.shape[0] > 16:
        selected = selected[:16]
    elif selected.shape[0] < 16:
        pad_slices = 16 - selected.shape[0]
        padding = np.zeros((pad_slices, selected.shape[1], selected.shape[2]), dtype=selected.dtype)
        selected = np.concatenate([selected, padding], axis=0)
    return selected

def create_comparison_figure(full_vol, half_vol, file_name, output_path):
    """
    Create CLEAN comparison figure WITHOUT any titles, labels, or annotations
    Pure image output only
    """
    fig = plt.figure(figsize=Config.FIGSIZE)
    
    # GridSpec for Full (left side) - NO margins
    gs_full = GridSpec(Config.GRID_ROWS, Config.GRID_COLS, figure=fig, 
                       hspace=0.02, wspace=0.02, 
                       left=0.01, right=0.49, 
                       top=0.99, bottom=0.01)
    
    # GridSpec for Half (right side) - NO margins
    gs_half = GridSpec(Config.GRID_ROWS, Config.GRID_COLS, figure=fig, 
                       hspace=0.02, wspace=0.02, 
                       left=0.51, right=0.99, 
                       top=0.99, bottom=0.01)
    
    # Plot all 16 slices - NO titles, NO axes, NO labels
    for idx in range(16):
        row = idx // Config.GRID_COLS
        col = idx % Config.GRID_COLS
        
        # Full projection (left)
        ax_full = fig.add_subplot(gs_full[row, col])
        ax_full.imshow(full_vol[idx], cmap=Config.COLORMAP, vmin=0, vmax=1)
        ax_full.axis('off')  # Remove all axes
        
        # Half projection (right)
        ax_half = fig.add_subplot(gs_half[row, col])
        ax_half.imshow(half_vol[idx], cmap=Config.COLORMAP, vmin=0, vmax=1)
        ax_half.axis('off')  # Remove all axes
    
    # Save with NO padding, NO borders
    plt.savefig(output_path, dpi=Config.DPI, 
                bbox_inches='tight', 
                pad_inches=0,  # NO padding
                facecolor='white')
    plt.close()
    print(f"    ✓ Saved: {os.path.basename(output_path)}")

# =====================================================================
# MAIN PROCESSING
# =====================================================================

def main():
    print("\n" + "="*70)
    print("Reference Image Generator – CLEAN MODE (No Annotations)")
    print("="*70)
    
    # Get common DICOM files
    full_files = {f.name for f in Path(Config.TEST_FULL_DIR).glob("*.dcm")}
    gen_files = {f.name for f in Path(Config.TEST_HALF_DIR).glob("*.dcm")}
    common_files = sorted(full_files & gen_files)
    
    if not common_files:
        print("\n❌ No common files found between:")
        print(f"  Full: {Config.TEST_FULL_DIR}")
        print(f"  Generated: {Config.TEST_HALF_DIR}")
        return
    
    print(f"\n✅ Found {len(common_files)} common files.")
    files_to_process = common_files[:Config.MAX_FILES]
    print(f"Processing first {len(files_to_process)} files...")
    
    success_count = 0
    for i, file_name in enumerate(files_to_process, 1):
        print(f"\n[{i}/{len(files_to_process)}] Processing: {file_name}")
        try:
            full_path = os.path.join(Config.TEST_FULL_DIR, file_name)
            gen_path = os.path.join(Config.TEST_HALF_DIR, file_name)
            
            # Load volumes
            full_vol, ok1 = load_dicom_volume(full_path)
            gen_vol, ok2 = load_dicom_volume(gen_path)
            
            if not (ok1 and ok2):
                print(f"    ✗ Skipped: Failed to load")
                continue
            
            # Select slices
            full_slices = select_slices(full_vol, Config.START_SLICE, Config.END_SLICE)
            gen_slices = select_slices(gen_vol, Config.START_SLICE, Config.END_SLICE)
            
            # Generate clean comparison
            output_path = os.path.join(Config.OUTPUT_DIR, 
                                      f"reference_clean_{Path(file_name).stem}.png")
            create_comparison_figure(full_slices, gen_slices, file_name, output_path)
            success_count += 1
            
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Successfully generated: {success_count}/{len(files_to_process)} clean images")
    print(f"Output directory: {Config.OUTPUT_DIR}")
    print(f"\nNote: Images have NO titles, labels, scales, or annotations")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()