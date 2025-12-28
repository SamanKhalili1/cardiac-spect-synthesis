"""
FIXED Enhanced UNet3D Test Script - WITH ROBUST CNR & GLOBAL NORMALIZATION
==================================================================================
CRITICAL UPDATES:
- Replaced flawed fixed-ROI CNR with Dynamic Thresholding CNR.
- CNR now automatically finds the heart (signal) and background regardless of position.
- Uses standard medical physics CNR formula: |Mu_S - Mu_B| / Sigma_B
- Maintained all previous fixes (normalization, DICOM scaling, etc.)
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pydicom
from tqdm import tqdm
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# =====================================================================
# CONFIGURATION
# =====================================================================

class TestConfig:
    BASE_DIR = "E:\\Project\\Cardiac"
    PREPROCESSING_DIR = os.path.join(BASE_DIR, "1_Pre_alternating")
    RESULTS_DIR = os.path.join(BASE_DIR, "2_Results")
    GENERATED_DIR = os.path.join(BASE_DIR, "3_Generated")
    GLOBAL_STATS_FILE = os.path.join(BASE_DIR, "global_normalization_stats.json")
    
    BASE_CHANNELS = 16
    INPUT_SHAPE = (32, 64, 64)
    DROPOUT_RATE = 0.1
    
    TEST_FOLD = 3
    
    BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_overall_model.pth")
    
    RESIZE_ORDER = 0
    GRADIENT_SIGMA = 0.2
    SHARPEN_INFERENCE = 0.35
    USE_SHARPENING = True

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================================
# GLOBAL NORMALIZATION
# =====================================================================

def load_global_stats():
    stats_file = TestConfig.GLOBAL_STATS_FILE
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Missing global stats: {stats_file}")
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    logger.info("Loaded global stats: p1=%.2f, p99=%.2f", stats['p1'], stats['p99'])
    return stats

def normalize_with_global_stats(volume, global_stats):
    p_low = global_stats['p1']
    p_high = global_stats['p99']
    if p_high > p_low:
        normalized = np.clip((volume - p_low) / (p_high - p_low), 0, 1)
    else:
        v_min, v_max = np.min(volume), np.max(volume)
        normalized = (volume - v_min) / (v_max - v_min + 1e-8) if v_max > v_min else np.zeros_like(volume)
    return normalized.astype(np.float32)

# =====================================================================
# MODEL ARCHITECTURE
# =====================================================================

class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

class DualAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(ch)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(ch)
    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += r
        return F.relu(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.res = ResidualBlock(out_ch)
        self.attn = DualAttention(out_ch)
        self.drop = nn.Dropout3d(drop)
        self.pool = nn.MaxPool3d(2)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.res(x)
        x = self.attn(x)
        x = self.drop(x)
        return self.pool(x), x

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, drop=0.1):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv1 = nn.Conv3d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.res = ResidualBlock(out_ch)
        self.attn = DualAttention(out_ch)
        self.drop = nn.Dropout3d(drop)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.res(x)
        x = self.attn(x)
        x = self.drop(x)
        return x

class UNet3D(nn.Module):
    def __init__(self, base_ch=16, drop=0.1):
        super().__init__()
        self.enc1 = EncoderBlock(1, base_ch, drop)
        self.enc2 = EncoderBlock(base_ch, base_ch*2, drop)
        self.enc3 = EncoderBlock(base_ch*2, base_ch*4, drop)
        self.enc4 = EncoderBlock(base_ch*4, base_ch*8, drop)
        self.enc5 = EncoderBlock(base_ch*8, base_ch*16, drop)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_ch*16, base_ch*32, 3, padding=1, bias=False),
            nn.BatchNorm3d(base_ch*32),
            nn.ReLU(inplace=True),
            *[ResidualBlock(base_ch*32) for _ in range(6)],
            DualAttention(base_ch*32),
            nn.Dropout3d(drop)
        )

        self.dec5 = DecoderBlock(base_ch*32, base_ch*16, base_ch*16, drop)
        self.dec4 = DecoderBlock(base_ch*16, base_ch*8, base_ch*8, drop)
        self.dec3 = DecoderBlock(base_ch*8, base_ch*4, base_ch*4, drop)
        self.dec2 = DecoderBlock(base_ch*4, base_ch*2, base_ch*2, drop)
        self.dec1 = DecoderBlock(base_ch*2, base_ch, base_ch, drop)

        self.head5 = nn.Conv3d(base_ch*16, 1, 1)
        self.head4 = nn.Conv3d(base_ch*8, 1, 1)
        self.head3 = nn.Conv3d(base_ch*4, 1, 1)
        self.head2 = nn.Conv3d(base_ch*2, 1, 1)
        self.head1 = nn.Conv3d(base_ch, 1, 1)

        self.fusion = nn.Conv3d(5, 1, 1)
        self.refine = nn.Conv3d(1, 1, 3, padding=1)

    @staticmethod
    def sharpen(x, strength=0.5):
        ksize = 3
        c = ksize // 2
        kernel = torch.zeros(1, 1, ksize, ksize, ksize, device=x.device)
        for i in range(ksize):
            for j in range(ksize):
                for k in range(ksize):
                    dx, dy, dz = i - c, j - c, k - c
                    kernel[0,0,i,j,k] = np.exp(-(dx*dx + dy*dy + dz*dz) / (2 * TestConfig.GRADIENT_SIGMA**2))
        kernel /= kernel.sum()
        blurred = F.conv3d(x, kernel, padding=c)
        sharp = x + strength * (x - blurred)
        return torch.clamp(sharp, 0.0, 1.0)

    def forward(self, x):
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)
        x5, s5 = self.enc5(x4)

        b = self.bottleneck(x5)

        d5 = self.dec5(b, s5)
        d4 = self.dec4(d5, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        p5 = F.interpolate(self.head5(d5), size=TestConfig.INPUT_SHAPE, mode='nearest')
        p4 = F.interpolate(self.head4(d4), size=TestConfig.INPUT_SHAPE, mode='nearest')
        p3 = F.interpolate(self.head3(d3), size=TestConfig.INPUT_SHAPE, mode='nearest')
        p2 = F.interpolate(self.head2(d2), size=TestConfig.INPUT_SHAPE, mode='nearest')
        p1 = self.head1(d1)

        fused = torch.cat([p1, p2, p3, p4, p5], dim=1)
        out = self.fusion(fused)
        out = self.refine(out)
        out = torch.sigmoid(out)

        if TestConfig.USE_SHARPENING:
            out = self.sharpen(out, TestConfig.SHARPEN_INFERENCE)
        return out

# =====================================================================
# METRIC UTILITIES (IMPROVED)
# =====================================================================

class Metrics:
    @staticmethod
    def calculate_ssim(pred, tgt, win=5):
        pred = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
        tgt = torch.from_numpy(tgt).unsqueeze(0).unsqueeze(0).float()
        pad = win // 2
        mu1 = F.avg_pool3d(pred, win, 1, pad)
        mu2 = F.avg_pool3d(tgt, win, 1, pad)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2
        sig1 = F.avg_pool3d(pred ** 2, win, 1, pad) - mu1_sq
        sig2 = F.avg_pool3d(tgt ** 2, win, 1, pad) - mu2_sq
        sig12 = F.avg_pool3d(pred * tgt, win, 1, pad) - mu12
        c1, c2 = 0.0001, 0.0009
        ssim_map = ((2 * mu12 + c1) * (2 * sig12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sig1 + sig2 + c2))
        return torch.clamp(ssim_map.mean(), 0.0, 1.0).item()

    @staticmethod
    def calculate_psnr(pred, tgt):
        mse = np.mean((pred - tgt) ** 2)
        return 100.0 if mse < 1e-10 else float(np.clip(20 * np.log10(1.0 / (np.sqrt(mse) + 1e-8)), 0, 100))

    @staticmethod
    def calculate_mae(pred, tgt):
        return float(np.mean(np.abs(pred - tgt)))

    @staticmethod
    def calculate_me(pred, tgt):
        return float(np.mean(pred - tgt))

    @staticmethod
    def calculate_rmse(pred, tgt):
        return float(np.sqrt(np.mean((pred - tgt) ** 2)))

    @staticmethod
    def calculate_cnr(vol):
        """
        Calculates CNR using DYNAMIC THRESHOLDING.
        Superior to fixed-crop methods as it automatically locates the heart (signal)
        and true background regardless of patient positioning.

        Signal ROI: Voxels > 60% of peak intensity (Assumed Heart/Target)
        Background ROI: Voxels < 10% of peak intensity (Assumed Air/Background)
        Formula: CNR = |Mean_Signal - Mean_Background| / Std_Background
        """
        # 1. Dynamic ROI Selection based on intensity percentiles
        peak_val = np.max(vol)
        signal_mask = vol > (0.6 * peak_val)  # Select brightest region (heart)
        bg_mask = vol < (0.1 * peak_val)      # Select darkest region (background)

        # Safety Check: Ensure masks are not empty (e.g., in flat images)
        if not signal_mask.any() or not bg_mask.any():
            return 0.0

        # 2. Calculate Statistics
        mu_s = np.nanmean(vol[signal_mask])
        mu_b = np.nanmean(vol[bg_mask])
        sigma_b = np.nanstd(vol[bg_mask]) # Use only background noise for standard detectability

        # 3. Final Calculation with zero-division protection
        if sigma_b < 1e-7:
            return 0.0
            
        return float(abs(mu_s - mu_b) / sigma_b)

# =====================================================================
# UTIL FUNCTIONS
# =====================================================================

def load_dicom(file_path, shape=TestConfig.INPUT_SHAPE, stats=None):
    try:
        dcm = pydicom.dcmread(file_path, force=True)
        vol = dcm.pixel_array.astype(np.float32)
        if vol.ndim == 2:
            vol = np.expand_dims(vol, 0)
        if vol.shape != shape:
            vol = resize_volume(vol, shape)
        vol = normalize_with_global_stats(vol, stats) if stats else vol / (vol.max() + 1e-8)
        return vol, dcm, True
    except Exception as e:
        logger.error("Failed to load %s: %s", file_path, e)
        return np.zeros(shape, np.float32), None, False

def resize_volume(vol, shape):
    if vol.shape == shape:
        return vol
    out = np.zeros(shape, np.float32)
    ds = vol.shape[0] / shape[0]
    for i in range(shape[0]):
        src = int(i * ds)
        if src < vol.shape[0]:
            out[i] = resize(vol[src], shape[1:], order=TestConfig.RESIZE_ORDER, preserve_range=True, anti_aliasing=False, mode='edge')
    return out

def denormalize_volume(vol, stats):
    p1, p99 = stats['p1'], stats['p99']
    denorm = vol * (p99 - p1) + p1
    return np.clip(denorm, 0, 65535).astype(np.uint16)

def save_dicom(output_vol, original_dcm, output_path, filename, stats):
    try:
        vol_uint16 = denormalize_volume(output_vol, stats)
        ds = original_dcm.copy()
        ds.Rows, ds.Columns = vol_uint16.shape[1], vol_uint16.shape[2]
        ds.NumberOfFrames = vol_uint16.shape[0]
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.PixelData = vol_uint16.tobytes()
        output_file = Path(output_path) / filename
        ds.save_as(str(output_file), enforce_file_format=True)
        return str(output_file)
    except Exception as e:
        logger.error("DICOM save error: %s", e)
        return None

# =====================================================================
# INFERENCE & TEST MANAGER
# =====================================================================

class TestManager:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stats = load_global_stats()
        Path(TestConfig.GENERATED_DIR).mkdir(parents=True, exist_ok=True)
        self.model = self._load_model()
        self.viz_dir = Path(TestConfig.GENERATED_DIR) / "Visualizations"
        self.viz_dir.mkdir(exist_ok=True)

    def _load_model(self):
        model = UNet3D(TestConfig.BASE_CHANNELS, TestConfig.DROPOUT_RATE).to(self.device)
        ckpt = torch.load(TestConfig.BEST_MODEL_PATH, map_location=self.device, weights_only=True)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _get_test_files(self):
        fold_dir = Path(TestConfig.PREPROCESSING_DIR) / f"Fold_{TestConfig.TEST_FOLD}"
        inp_dir = fold_dir / "Test_Half"
        tgt_dir = fold_dir / "Test_Full"
        inp_files = {f.name: f for f in inp_dir.glob("*.dcm")}
        tgt_files = {f.name: f for f in tgt_dir.glob("*.dcm")}
        common = sorted(set(inp_files) & set(tgt_files))
        return [(str(inp_files[f]), str(tgt_files[f]), f) for f in common]

    def run_test(self):
        test_files = self._get_test_files()
        if not test_files:
            raise ValueError("No test files found!")
        
        logger.info(f"Testing Fold {TestConfig.TEST_FOLD} with {len(test_files)} files")

        all_results = []
        metric_lists = {'ssim': [], 'psnr': [], 'cnr': [], 'rmse': [], 'mae': [], 'me': []}

        for inp_path, tgt_path, fname in tqdm(test_files, desc="Testing"):
            inp, _, _ = load_dicom(inp_path, TestConfig.INPUT_SHAPE, self.stats)
            tgt, dcm, _ = load_dicom(tgt_path, TestConfig.INPUT_SHAPE, self.stats)
            
            with torch.no_grad():
                inp_tensor = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).float().to(self.device)
                out_tensor = self.model(inp_tensor)
                out = out_tensor.squeeze().cpu().numpy()

            metrics = {
                'ssim': Metrics.calculate_ssim(out, tgt),
                'psnr': Metrics.calculate_psnr(out, tgt),
                'cnr': Metrics.calculate_cnr(out),  # Now uses robust dynamic method
                'rmse': Metrics.calculate_rmse(out, tgt),
                'mae': Metrics.calculate_mae(out, tgt),
                'me': Metrics.calculate_me(out, tgt)
            }

            save_dicom(out, dcm, TestConfig.GENERATED_DIR, fname, self.stats)
            self._save_viz(inp, tgt, out, fname)

            all_results.append({'filename': fname, 'metrics': metrics})
            for k, v in metrics.items():
                metric_lists[k].append(v)

        self._save_summary(all_results, metric_lists)

    def _save_viz(self, inp, tgt, out, fname):
        slice_idx = inp.shape[0] // 2
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        cmap = 'jet'
        vmin, vmax = 0.0, 1.0
        
        for ax, data, title in zip(axes, [inp, tgt, out], ['Input', 'Target', 'Output']):
            im = ax.imshow(data[slice_idx], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f"{Path(fname).stem}.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _save_summary(self, results, metric_lists):
        stats_summary = {}
        for metric, values in metric_lists.items():
            stats_summary[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }

        output = {
            'test_info': {
                'fold': TestConfig.TEST_FOLD,
                'files': len(results),
                'date': datetime.now().isoformat(),
                'model': str(TestConfig.BEST_MODEL_PATH)
            },
            'global_stats_used': self.stats,
            'statistics': stats_summary,
            'individual_results': results
        }

        # Save JSON
        json_path = Path(TestConfig.GENERATED_DIR) / "test_results.json"
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)

        # Save TXT
        txt_path = Path(TestConfig.GENERATED_DIR) / "test_summary.txt"
        with open(txt_path, 'w') as f:
            f.write("Test Summary\n")
            f.write("="*50 + "\n")
            for metric, stat in stats_summary.items():
                f.write(f"{metric.upper():<6}: mean={stat['mean']:.6f}, std={stat['std']:.6f}, "
                        f"min={stat['min']:.6f}, max={stat['max']:.6f}, median={stat['median']:.6f}\n")

        logger.info(f"Results saved to {TestConfig.GENERATED_DIR}")

# =====================================================================
# MAIN
# =====================================================================

def main():
    print("Running Enhanced UNet3D Test with ROBUST CNR Metrics")
    TestManager().run_test()
    print("✅ Test completed successfully!")

if __name__ == "__main__":
    main()

