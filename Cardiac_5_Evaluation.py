"""
Evaluation Script for Cardiac SPECT Reconstruction
Compares volumes in 4_Recon_Generated vs 4_Recon_Full
Computes: PSNR, SSIM, RMSE, MAE, ME, CNR
Saves results to E:\Project\Cardiac\5_Evaluation
"""

import os
import json
import numpy as np
import pydicom
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# Configuration
# =====================================================================

class EvalConfig:
    BASE_DIR = r"E:\Project\Cardiac"
    RECON_FULL_DIR = os.path.join(BASE_DIR, "4_Recon_Full")
    RECON_GEN_DIR = os.path.join(BASE_DIR, "4_Recon_Generated")
    OUTPUT_DIR = os.path.join(BASE_DIR, "5_Evaluation")
    
    # Ensure consistent shape with training
    EXPECTED_SHAPE = (32, 64, 64)  # Matches UNet3D output
    
    # SSIM settings
    SSIM_WINDOW_SIZE = 5

# =====================================================================
# Utility Functions
# =====================================================================

def load_dicom_volume(file_path):
    """Load DICOM volume, return float32 numpy array"""
    try:
        ds = pydicom.dcmread(file_path, force=True)
        if hasattr(ds, 'pixel_array'):
            vol = ds.pixel_array.astype(np.float32)
            if vol.ndim == 3 and vol.shape[0] == 1:
                vol = vol[0]
            return vol, True
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
    return np.zeros(EvalConfig.EXPECTED_SHAPE, dtype=np.float32), False

def calculate_psnr(pred, target):
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(np.clip(20 * np.log10(1.0 / np.sqrt(mse)), 0, 100))

def calculate_ssim_3d(pred, target, win=5):
    from torch.nn.functional import avg_pool3d
    import torch
    pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
    target_t = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).float()
    pad = win // 2
    mu1 = avg_pool3d(pred_t, win, 1, pad)
    mu2 = avg_pool3d(target_t, win, 1, pad)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2
    sig1 = avg_pool3d(pred_t ** 2, win, 1, pad) - mu1_sq
    sig2 = avg_pool3d(target_t ** 2, win, 1, pad) - mu2_sq
    sig12 = avg_pool3d(pred_t * target_t, win, 1, pad) - mu12
    c1, c2 = 0.0001, 0.0009
    ssim_map = ((2 * mu12 + c1) * (2 * sig12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sig1 + sig2 + c2))
    return float(torch.clamp(ssim_map.mean(), 0.0, 1.0).item())

def calculate_cnr(vol):
    """Robust CNR using dynamic intensity thresholds"""
    peak = np.max(vol)
    if peak == 0:
        return 0.0
    signal_mask = vol > (0.6 * peak)
    bg_mask = vol < (0.1 * peak)
    if not np.any(signal_mask) or not np.any(bg_mask):
        return 0.0
    mu_s = np.mean(vol[signal_mask])
    mu_b = np.mean(vol[bg_mask])
    sigma_b = np.std(vol[bg_mask])
    if sigma_b < 1e-7:
        return 0.0
    return float(abs(mu_s - mu_b) / sigma_b)

def calculate_rmse(pred, target):
    return float(np.sqrt(np.mean((pred - target) ** 2)))

def calculate_mae(pred, target):
    return float(np.mean(np.abs(pred - target)))

def calculate_me(pred, target):
    return float(np.mean(pred - target))

# =====================================================================
# Main Evaluator
# =====================================================================

def main():
    os.makedirs(EvalConfig.OUTPUT_DIR, exist_ok=True)
    
    # Get common filenames
    full_files = {f.name: f for f in Path(EvalConfig.RECON_FULL_DIR).glob("*.dcm")}
    gen_files = {f.name: f for f in Path(EvalConfig.RECON_GEN_DIR).glob("*.dcm")}
    common_names = sorted(set(full_files.keys()) & set(gen_files.keys()))
    
    if not common_names:
        print("[ERROR] No matching files found between Full and Generated directories!")
        return
    
    print(f"[INFO] Found {len(common_names)} matching cases.")
    
    results = []
    metrics = {'psnr': [], 'ssim': [], 'rmse': [], 'mae': [], 'me': [], 'cnr': []}
    
    for fname in tqdm(common_names, desc="Evaluating"):
        full_path = full_files[fname]
        gen_path = gen_files[fname]
        
        full_vol, ok1 = load_dicom_volume(full_path)
        gen_vol, ok2 = load_dicom_volume(gen_path)
        
        if not (ok1 and ok2):
            print(f"[SKIP] Failed to load {fname}")
            continue
        
        # Ensure shape consistency
        if full_vol.shape != EvalConfig.EXPECTED_SHAPE or gen_vol.shape != EvalConfig.EXPECTED_SHAPE:
            print(f"[SKIP] Shape mismatch in {fname}: full={full_vol.shape}, gen={gen_vol.shape}")
            continue
        
        # Normalize to [0,1] for fair comparison (since scaling may differ)
        def normalize_01(x):
            x = x.astype(np.float32)
            x_min, x_max = x.min(), x.max()
            if x_max > x_min:
                return (x - x_min) / (x_max - x_min)
            return np.zeros_like(x)
        
        full_norm = normalize_01(full_vol)
        gen_norm = normalize_01(gen_vol)
        
        # Compute metrics
        ssim_val = calculate_ssim_3d(gen_norm, full_norm, EvalConfig.SSIM_WINDOW_SIZE)
        psnr_val = calculate_psnr(gen_norm, full_norm)
        rmse_val = calculate_rmse(gen_norm, full_norm)
        mae_val = calculate_mae(gen_norm, full_norm)
        me_val = calculate_me(gen_norm, full_norm)
        cnr_val = calculate_cnr(gen_norm)  # CNR on generated (as in test script)
        
        result = {
            'filename': fname,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'rmse': rmse_val,
            'mae': mae_val,
            'me': me_val,
            'cnr': cnr_val
        }
        results.append(result)
        
        for k, v in result.items():
            if k != 'filename':
                metrics[k].append(v)
    
    # Compute statistics
    stats = {}
    for metric, values in metrics.items():
        if values:
            stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        else:
            stats[metric] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
    
    # Save JSON
    output_json = os.path.join(EvalConfig.OUTPUT_DIR, "evaluation_results.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({
            'evaluation_info': {
                'timestamp': datetime.now().isoformat(),
                'full_dir': EvalConfig.RECON_FULL_DIR,
                'generated_dir': EvalConfig.RECON_GEN_DIR,
                'num_cases': len(results)
            },
            'statistics': stats,
            'individual_results': results
        }, f, indent=2, ensure_ascii=False)
    
    # Save TXT
    output_txt = os.path.join(EvalConfig.OUTPUT_DIR, "evaluation_summary.txt")
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("Cardiac SPECT Reconstruction Evaluation\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Cases evaluated: {len(results)}\n\n")
        for metric in ['ssim', 'psnr', 'cnr', 'rmse', 'mae', 'me']:
            s = stats[metric]
            f.write(f"{metric.upper():<6}: mean={s['mean']:.6f}, std={s['std']:.6f}, "
                    f"min={s['min']:.6f}, max={s['max']:.6f}, median={s['median']:.6f}\n")
    
    print(f"\n✅ Evaluation completed!")
    print(f"   Results saved to: {EvalConfig.OUTPUT_DIR}")
    print(f"   Summary:\n")
    for metric in ['SSIM', 'PSNR', 'CNR']:
        s = stats[metric.lower()]
        print(f"   {metric}: {s['mean']:.4f} ± {s['std']:.4f}  [median: {s['median']:.4f}]")

if __name__ == "__main__":
    main()

