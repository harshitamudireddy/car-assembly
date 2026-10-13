import os
import cv2
from bead_engine import BeadInspector

inspector = BeadInspector()
samples = sorted(os.listdir('test_samples'))

print("--- BENCHMARKING BEAD INSPECTOR ON TEST SAMPLES ---")
for s in samples:
    path = os.path.join('test_samples', s)
    img = cv2.imread(path)
    res, anno = inspector.inspect(img)
    print(f"[{s}]")
    print(f"  Verdict: {res['verdict']}")
    print(f"  Coverage: {res['overall_coverage_pct']}% | Placement: {res['overall_placement_pct']}%")
    for z in res['zones']:
        print(f"    - {z['zone']}: {z['status']} (cov={z['coverage_pct']}%, acc={z['placement_accuracy_pct']}%, gaps={z['gap_count']})")
    
    # Save annotated test image to dedicated output folder
    out_dir = 'inspection_results'
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"inspected_{s}"), anno)
print("\nEvaluation complete! Annotated results saved to inspection_results/.")
