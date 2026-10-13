"""
Automotive Sealant Bead Quality Verification Engine
Provides deterministic, industrial-grade tolerance corridor inspection,
continuity/gap detection, and presence verification.
"""

import os
import cv2
import numpy as np


class BeadInspector:
    def __init__(self, base_dir=None, tolerance_px=18, min_coverage=0.85, max_misplaced_ratio=0.15):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir
        self.tolerance_px = tolerance_px
        self.min_coverage = min_coverage
        self.max_misplaced_ratio = max_misplaced_ratio

        # Load golden masks
        mask_all_path = os.path.join(base_dir, 'golden_mask.png')
        mask_cup_path = os.path.join(base_dir, 'golden_cup_mask.png')
        mask_flange_path = os.path.join(base_dir, 'golden_flange_mask.png')
        mask_cutout_path = os.path.join(base_dir, 'golden_cutout_mask.png')

        if not os.path.exists(mask_all_path):
            raise FileNotFoundError(f"Golden mask not found at {mask_all_path}. Run calibration first.")

        self.golden_all = cv2.imread(mask_all_path, cv2.IMREAD_GRAYSCALE)
        self.golden_cup = cv2.imread(mask_cup_path, cv2.IMREAD_GRAYSCALE)
        self.golden_flange = cv2.imread(mask_flange_path, cv2.IMREAD_GRAYSCALE)
        self.golden_cutout = cv2.imread(mask_cutout_path, cv2.IMREAD_GRAYSCALE)

        self.h, self.w = self.golden_all.shape[:2]

        # Define Sheet Metal Inspection Region of Interest (strictly part surfaces)
        self.roi_metal = np.zeros((self.h, self.w), dtype=np.uint8)
        
        # 1. Cup Bracket ROI
        cv2.ellipse(self.roi_metal, (226, 354), (55, 45), -18, 0, 360, 255, -1)

        # 2. Main stamping body (excluding clamps and upper bracket hardware)
        part_poly = np.array([
            [285, 310], [630, 280], [635, 95], [920, 160], [920, 260],
            [920, 360], [920, 680], [420, 680], [285, 680]
        ], np.int32)
        cv2.fillPoly(self.roi_metal, [part_poly], 255)

        # Cutout the center hole (pipe/background reflection inside aperture)
        inner_hole = np.array([
            [580, 350], [830, 370], [840, 590], [580, 590]
        ], np.int32)
        cv2.fillPoly(self.roi_metal, [inner_hole], 0)

        # Precompute Tolerance Corridors (Dilated regions around nominal paths)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.tolerance_px * 2 + 1, self.tolerance_px * 2 + 1))
        self.corridor_all = cv2.dilate(self.golden_all, kernel)
        self.corridor_cup = cv2.dilate(self.golden_cup, kernel)
        self.corridor_flange = cv2.dilate(self.golden_flange, kernel)
        self.corridor_cutout = cv2.dilate(self.golden_cutout, kernel)

    def extract_yellow_bead(self, img):
        """Extract yellow paste segmentation mask within part ROI."""
        if img.shape[:2] != (self.h, self.w):
            img = cv2.resize(img, (self.w, self.h))

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([16, 85, 85])
        upper_yellow = np.array([33, 255, 255])
        raw_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Restrict strictly to sheet metal ROI
        bead_mask = cv2.bitwise_and(raw_yellow, raw_yellow, mask=self.roi_metal)

        # Remove speckle noise
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bead_mask = cv2.morphologyEx(bead_mask, cv2.MORPH_OPEN, k)

        # Remove isolated components smaller than 25 pixels
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bead_mask)
        clean_mask = np.zeros_like(bead_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 25:
                clean_mask[labels == i] = 255

        return clean_mask

    def inspect_zone(self, detected_mask, golden_zone_mask, corridor_zone_mask, zone_name):
        """
        Evaluates presence, placement compliance, and gaps for a specific bead zone.
        """
        golden_area = np.count_nonzero(golden_zone_mask)
        if golden_area == 0:
            return {}

        # 1. Presence: yellow inside the golden path / corridor
        inside_corridor = cv2.bitwise_and(detected_mask, corridor_zone_mask)
        inside_nominal = cv2.bitwise_and(detected_mask, golden_zone_mask)
        
        inside_corridor_area = np.count_nonzero(inside_corridor)
        inside_nominal_area = np.count_nonzero(inside_nominal)

        coverage_ratio = inside_corridor_area / max(golden_area, 1)
        coverage_pct = round(min(coverage_ratio, 1.0) * 100.0, 1)

        # 2. Placement: yellow in this zone bounding region that is OUTSIDE the tolerance corridor
        ys, xs = np.where(corridor_zone_mask > 0)
        if len(ys) > 0:
            ymin, ymax = max(0, ys.min() - 35), min(self.h, ys.max() + 35)
            xmin, xmax = max(0, xs.min() - 35), min(self.w, xs.max() + 35)
            zone_roi = np.zeros_like(detected_mask)
            zone_roi[ymin:ymax, xmin:xmax] = 255
            detected_in_zone = cv2.bitwise_and(detected_mask, zone_roi)
        else:
            detected_in_zone = detected_mask

        total_zone_detected = np.count_nonzero(detected_in_zone)
        outside_corridor = cv2.bitwise_and(detected_in_zone, cv2.bitwise_not(corridor_zone_mask))
        outside_area = np.count_nonzero(outside_corridor)

        misplaced_ratio = outside_area / max(total_zone_detected, 1) if total_zone_detected > 0 else 0.0
        placement_accuracy_pct = round(max(0.0, (1.0 - misplaced_ratio) * 100.0), 1)

        # 3. Continuity / Gap Detection
        missing_mask = cv2.bitwise_and(golden_zone_mask, cv2.bitwise_not(detected_mask))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        missing_clean = cv2.morphologyEx(missing_mask, cv2.MORPH_OPEN, k)
        num_gaps, _, stats_gaps, _ = cv2.connectedComponentsWithStats(missing_clean)
        significant_gaps = 0
        for i in range(1, num_gaps):
            if stats_gaps[i, cv2.CC_STAT_AREA] >= 35:
                significant_gaps += 1

        # Verdict logic
        if coverage_pct < 25.0:
            status = "MISSING"
            is_pass = False
        elif coverage_pct < (self.min_coverage * 100.0):
            status = "INCOMPLETE"
            is_pass = False
        elif misplaced_ratio > self.max_misplaced_ratio:
            status = "MISPLACED"
            is_pass = False
        elif significant_gaps > 0 and coverage_pct < 95.0:
            status = "GAP_DEFECT"
            is_pass = False
        else:
            status = "PASS"
            is_pass = True

        return {
            "zone": zone_name,
            "status": status,
            "is_pass": is_pass,
            "coverage_pct": coverage_pct,
            "placement_accuracy_pct": placement_accuracy_pct,
            "gap_count": significant_gaps,
            "detected_pixels": total_zone_detected,
            "outside_pixels": outside_area
        }

    def inspect(self, img):
        """
        Full inspection pipeline for an image.
        Returns:
            result_dict: Comprehensive metrics and status
            annotated_img: Visual overlay image
        """
        if img.shape[:2] != (self.h, self.w):
            img = cv2.resize(img, (self.w, self.h))

        detected_mask = self.extract_yellow_bead(img)

        # Evaluate the 3 zones
        res_cup = self.inspect_zone(detected_mask, self.golden_cup, self.corridor_cup, "Zone 1: Bracket Cup")
        res_flange = self.inspect_zone(detected_mask, self.golden_flange, self.corridor_flange, "Zone 2: Upper Flange")
        res_cutout = self.inspect_zone(detected_mask, self.golden_cutout, self.corridor_cutout, "Zone 3: Center Cutout")

        zone_results = [res_cup, res_flange, res_cutout]

        # Overall presence and placement
        total_golden_area = np.count_nonzero(self.golden_all)
        total_detected_area = np.count_nonzero(detected_mask)
        inside_corridor = cv2.bitwise_and(detected_mask, self.corridor_all)
        inside_corridor_area = np.count_nonzero(inside_corridor)
        outside_corridor = cv2.bitwise_and(detected_mask, cv2.bitwise_not(self.corridor_all))
        outside_area = np.count_nonzero(outside_corridor)

        overall_coverage_pct = round(min(inside_corridor_area / max(total_golden_area, 1), 1.0) * 100.0, 1)
        overall_placement_pct = round((inside_corridor_area / max(total_detected_area, 1)) * 100.0, 1) if total_detected_area > 0 else 0.0

        # Overall Status
        all_passed = all(z["is_pass"] for z in zone_results)
        if all_passed:
            overall_verdict = "PASS"
            verdict_color = (0, 200, 0)
        elif total_detected_area < 800 or overall_coverage_pct < 10.0:
            overall_verdict = "FAIL - NO PASTE PRESENT"
            verdict_color = (0, 0, 230)
        elif any(z["status"] == "MISPLACED" for z in zone_results) or (outside_area / max(total_detected_area, 1) > 0.20):
            overall_verdict = "FAIL - PASTE MISPLACED"
            verdict_color = (0, 140, 255)
        elif any(z["status"] == "GAP_DEFECT" for z in zone_results):
            overall_verdict = "FAIL - DISCONTINUITY / GAP"
            verdict_color = (0, 0, 230)
        else:
            failed_zones = [z["zone"].split(":")[1].strip() for z in zone_results if not z["is_pass"]]
            overall_verdict = f"FAIL - INCOMPLETE ({', '.join(failed_zones)})"
            verdict_color = (0, 0, 230)

        # Generate Visual Annotation Overlay
        annotated = img.copy()

        # 1. Subtle Tolerance Corridor outline (Cyan)
        corridor_contours, _ = cv2.findContours(self.corridor_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, corridor_contours, -1, (255, 200, 0), 1)

        # 2. Alpha-blended Bead Overlay (Green = In tolerance, Orange = Misplaced)
        overlay = annotated.copy()
        if inside_corridor_area > 0:
            overlay[inside_corridor > 0] = [0, 255, 0]
        if outside_area > 0:
            overlay[outside_corridor > 0] = [0, 100, 255]

        annotated = cv2.addWeighted(annotated, 0.45, overlay, 0.55, 0)

        # 4. Missing bead locations (Red outline along missing nominal path)
        missing_mask = cv2.bitwise_and(self.golden_all, cv2.bitwise_not(detected_mask))
        missing_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        missing_clean = cv2.morphologyEx(missing_mask, cv2.MORPH_OPEN, missing_k)
        missing_contours, _ = cv2.findContours(missing_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in missing_contours:
            if cv2.contourArea(c) > 30:
                cv2.drawContours(annotated, [c], -1, (0, 0, 255), 2)

        # 5. Draw HUD Banner on top
        cv2.rectangle(annotated, (20, 20), (620, 150), (20, 20, 20), -1)
        cv2.rectangle(annotated, (20, 20), (620, 150), verdict_color, 2)

        cv2.putText(annotated, overall_verdict, (35, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, verdict_color, 2)
        cv2.putText(annotated, f"Coverage: {overall_coverage_pct}%  |  Placement Accuracy: {overall_placement_pct}%",
                    (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(annotated, f"Bead Pixels: {total_detected_area}  |  Corridor Margin: +/-{self.tolerance_px}px",
                    (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1)

        result_dict = {
            "verdict": overall_verdict,
            "is_pass": all_passed,
            "overall_coverage_pct": overall_coverage_pct,
            "overall_placement_pct": overall_placement_pct,
            "total_detected_pixels": total_detected_area,
            "outside_tolerance_pixels": outside_area,
            "zones": zone_results
        }

        return result_dict, annotated
