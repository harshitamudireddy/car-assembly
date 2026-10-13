"""
Synthetic Dataset Generator for Automotive Sealant Bead Inspection.
Creates photorealistic defect variations (Pass, Missing, Gaps, Misplaced)
using OpenCV inpainting and morphological defect injection.
"""

import os
import random
import cv2
import numpy as np


def apply_photometric_jitter(img):
    """Simulate industrial lighting shifts, slight glare, and sensor noise."""
    out = img.astype(np.float32)

    # Brightness & Contrast
    alpha = random.uniform(0.85, 1.15) # Contrast
    beta = random.uniform(-18.0, 18.0) # Brightness
    out = out * alpha + beta

    # Color balance jitter (minor Kelvin shift)
    for c in range(3):
        out[:, :, c] *= random.uniform(0.95, 1.05)

    out = np.clip(out, 0, 255).astype(np.uint8)

    # Sensor noise
    if random.random() > 0.4:
        noise = np.random.normal(0, random.uniform(1.5, 4.0), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Slight camera defocus
    if random.random() > 0.5:
        ksize = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (ksize, ksize), 0)

    return out


def apply_geometric_jitter(img, mask_paste=None, max_shift=6, max_rot=2.0):
    """Simulate minor fixture mechanical clamping tolerances."""
    h, w = img.shape[:2]
    angle = random.uniform(-max_rot, max_rot)
    tx = random.uniform(-max_shift, max_shift)
    ty = random.uniform(-max_shift, max_shift)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty

    img_warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    if mask_paste is not None:
        mask_warped = cv2.warpAffine(mask_paste, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img_warped, mask_warped
    return img_warped


def blend_paste_on_metal(bare_metal, golden_img, paste_mask):
    """Blends genuine yellow paste texture onto bare metal using the specified paste mask."""
    result = bare_metal.copy()
    result[paste_mask > 0] = golden_img[paste_mask > 0]
    return result


def create_gap_defect(paste_mask, num_gaps=2, gap_size=40):
    """Punch gaps / breaks along the bead path."""
    out_mask = paste_mask.copy()
    ys, xs = np.where(paste_mask > 0)
    if len(ys) == 0:
        return out_mask

    for _ in range(num_gaps):
        idx = random.randint(0, len(ys) - 1)
        cy, cx = ys[idx], xs[idx]
        cv2.circle(out_mask, (cx, cy), gap_size, 0, -1)

    return out_mask


def create_misplaced_defect(bare_metal, golden_img, paste_mask, zone="random"):
    """Shifts or warps beads outside the nominal stamped groove."""
    h, w = paste_mask.shape[:2]
    misplaced_mask = np.zeros_like(paste_mask)

    # Pick shift direction
    shift_x = random.choice([-1, 1]) * random.randint(25, 45)
    shift_y = random.choice([-1, 1]) * random.randint(20, 40)

    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_mask = cv2.warpAffine(paste_mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Paste the shifted bead
    result = bare_metal.copy()
    # Sample bead color/texture
    shifted_pixels = cv2.warpAffine(golden_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    result[shifted_mask > 0] = shifted_pixels[shifted_mask > 0]

    return result


def generate_dataset(base_dir=None, num_train_per_class=40, num_val_per_class=10):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    golden_img = cv2.imread(os.path.join(base_dir, 'WhatsApp Image 2026-10-13 at 19.11.31.jpeg'))
    bare_metal = cv2.imread(os.path.join(base_dir, 'golden_bare_metal.jpg'))
    golden_all = cv2.imread(os.path.join(base_dir, 'golden_mask.png'), cv2.IMREAD_GRAYSCALE)
    golden_cup = cv2.imread(os.path.join(base_dir, 'golden_cup_mask.png'), cv2.IMREAD_GRAYSCALE)
    golden_flange = cv2.imread(os.path.join(base_dir, 'golden_flange_mask.png'), cv2.IMREAD_GRAYSCALE)
    golden_cutout = cv2.imread(os.path.join(base_dir, 'golden_cutout_mask.png'), cv2.IMREAD_GRAYSCALE)

    classes = ["PASS_CORRECT", "FAIL_MISSING", "FAIL_GAP_DEFECT", "FAIL_MISPLACED"]
    dataset_dir = os.path.join(base_dir, 'dataset')
    test_samples_dir = os.path.join(base_dir, 'test_samples')

    for split in ['train', 'val']:
        for cls_name in classes:
            os.makedirs(os.path.join(dataset_dir, split, cls_name), exist_ok=True)
    os.makedirs(test_samples_dir, exist_ok=True)

    print("Generating training and validation datasets...")

    for split, count in [('train', num_train_per_class), ('val', num_val_per_class)]:
        for i in range(count):
            # 1. PASS_CORRECT
            mask_p = golden_all.copy()
            if random.random() > 0.5:
                # Slight width variation
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_p = cv2.dilate(mask_p, k) if random.random() > 0.5 else cv2.erode(mask_p, k)
            img_pass = blend_paste_on_metal(bare_metal, golden_img, mask_p)
            img_pass = apply_geometric_jitter(img_pass)
            img_pass = apply_photometric_jitter(img_pass)
            cv2.imwrite(os.path.join(dataset_dir, split, "PASS_CORRECT", f"pass_{i:04d}.jpg"), img_pass)

            # 2. FAIL_MISSING
            missing_mode = random.choice(["all", "cup", "flange", "cutout", "cup_flange"])
            if missing_mode == "all":
                mask_m = np.zeros_like(golden_all)
            elif missing_mode == "cup":
                mask_m = golden_flange | golden_cutout
            elif missing_mode == "flange":
                mask_m = golden_cup | golden_cutout
            elif missing_mode == "cutout":
                mask_m = golden_cup | golden_flange
            else:
                mask_m = golden_cutout

            img_missing = blend_paste_on_metal(bare_metal, golden_img, mask_m)
            img_missing = apply_geometric_jitter(img_missing)
            img_missing = apply_photometric_jitter(img_missing)
            cv2.imwrite(os.path.join(dataset_dir, split, "FAIL_MISSING", f"missing_{i:04d}.jpg"), img_missing)

            # 3. FAIL_GAP_DEFECT
            target_mask = random.choice([golden_cutout, golden_flange, golden_all])
            gapped = create_gap_defect(target_mask, num_gaps=random.randint(1, 3), gap_size=random.randint(25, 45))
            full_gap_mask = (golden_all & ~target_mask) | gapped
            img_gap = blend_paste_on_metal(bare_metal, golden_img, full_gap_mask)
            img_gap = apply_geometric_jitter(img_gap)
            img_gap = apply_photometric_jitter(img_gap)
            cv2.imwrite(os.path.join(dataset_dir, split, "FAIL_GAP_DEFECT", f"gap_{i:04d}.jpg"), img_gap)

            # 4. FAIL_MISPLACED
            misplace_target = random.choice([golden_cutout, golden_flange, golden_all])
            img_misplaced = create_misplaced_defect(bare_metal, golden_img, misplace_target)
            img_misplaced = apply_photometric_jitter(img_misplaced)
            cv2.imwrite(os.path.join(dataset_dir, split, "FAIL_MISPLACED", f"misplaced_{i:04d}.jpg"), img_misplaced)

    # Generate dedicated demo showcase test cases
    print("Generating dedicated test samples for dashboard demo...")
    # Sample 1: Perfect Golden Pass
    cv2.imwrite(os.path.join(test_samples_dir, "01_PASS_Golden_Nominal.jpg"), golden_img)

    # Sample 2: Complete Paste Missing
    cv2.imwrite(os.path.join(test_samples_dir, "02_FAIL_Missing_All_Paste.jpg"), bare_metal)

    # Sample 3: Missing Cup Bead
    mask_no_cup = golden_flange | golden_cutout
    cv2.imwrite(os.path.join(test_samples_dir, "03_FAIL_Missing_Cup_Bead.jpg"), blend_paste_on_metal(bare_metal, golden_img, mask_no_cup))

    # Sample 4: Missing Flange Bead
    mask_no_flange = golden_cup | golden_cutout
    cv2.imwrite(os.path.join(test_samples_dir, "04_FAIL_Missing_Flange_Bead.jpg"), blend_paste_on_metal(bare_metal, golden_img, mask_no_flange))

    # Sample 5: Cutout Gap Defect (Nozzle Clog)
    cutout_gap = create_gap_defect(golden_cutout, num_gaps=2, gap_size=35)
    mask_gap = golden_cup | golden_flange | cutout_gap
    cv2.imwrite(os.path.join(test_samples_dir, "05_FAIL_Cutout_Gap_Defect.jpg"), blend_paste_on_metal(bare_metal, golden_img, mask_gap))

    # Sample 6: Misplaced Flange Bead
    flange_misplaced = create_misplaced_defect(bare_metal, golden_img, golden_flange)
    mask_ok = golden_cup | golden_cutout
    flange_misplaced[mask_ok > 0] = golden_img[mask_ok > 0]
    cv2.imwrite(os.path.join(test_samples_dir, "06_FAIL_Misplaced_Flange_Bead.jpg"), flange_misplaced)

    # Sample 7: Misplaced Cutout Loop Bead
    cutout_misplaced = create_misplaced_defect(bare_metal, golden_img, golden_cutout)
    mask_ok2 = golden_cup | golden_flange
    cutout_misplaced[mask_ok2 > 0] = golden_img[mask_ok2 > 0]
    cv2.imwrite(os.path.join(test_samples_dir, "07_FAIL_Misplaced_Cutout_Loop.jpg"), cutout_misplaced)

    print(f"Dataset successfully created in {dataset_dir}!")
    print(f"Test samples created in {test_samples_dir}!")


if __name__ == '__main__':
    generate_dataset()
