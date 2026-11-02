"""
Generates visual inspection images for each preprocessing step:
- Raw contactless fingerphoto
- K-Means background segmentation mask
- Cropped ROI
- Adaptive Gaussian Thresholding
- Multi-orientation Gabor Filter Enhancement
- Thinned Skeleton
- Minutiae Overlay (Terminations in red, Bifurcations in blue)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from verifnet.preprocessing import FingerprintPreprocessor
from verifnet.minutiae_matcher import MinutiaeMatcher


def visualize_pipeline(sample_path, save_path="verifnet/preprocessing_steps.png"):
    img = cv2.imread(sample_path)
    prep = FingerprintPreprocessor(target_size=(256, 256))
    matcher = MinutiaeMatcher(border_margin=15, min_dist=6)

    res = prep.process(img)
    minutiae = matcher.extract_minutiae(res['skeleton'], res['mask'], res['dist_map'])

    # Create Minutiae overlay on enhanced image
    overlay = cv2.cvtColor(res['enhanced_uint8'], cv2.COLOR_GRAY2BGR)
    term_count = 0
    bif_count = 0
    for m in minutiae:
        x, y, theta, cn = int(m[0]), int(m[1]), m[2], int(m[3])
        if cn == 1:
            # Termination: Red circle
            cv2.circle(overlay, (x, y), 3, (0, 0, 255), 1)
            # Short direction line
            dx = int(6 * np.cos(theta))
            dy = int(6 * np.sin(theta))
            cv2.line(overlay, (x, y), (x + dx, y + dy), (0, 0, 255), 1)
            term_count += 1
        elif cn == 3:
            # Bifurcation: Blue circle
            cv2.circle(overlay, (x, y), 3, (255, 0, 0), 1)
            dx = int(6 * np.cos(theta))
            dy = int(6 * np.sin(theta))
            cv2.line(overlay, (x, y), (x + dx, y + dy), (255, 0, 0), 1)
            bif_count += 1

    # Plot grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("(a) Input Contactless Fingerphoto", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(res['mask'], cmap='gray')
    axes[0, 1].set_title("(b) K-Means Masking", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(res['adaptive_thresh'], cmap='gray')
    axes[0, 2].set_title("(c) Adaptive Gaussian Thresholding", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(res['enhanced_uint8'], cmap='gray')
    axes[1, 0].set_title("(d) Gabor Filter Enhancement", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(res['skeleton'], cmap='gray')
    axes[1, 1].set_title("(e) Thinned Skeleton", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f"(f) Minutiae ({term_count} Terminations [Red], {bif_count} Bifurcations [Blue])",
                          fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    plt.suptitle("VerifNet Preprocessing & Feature Extraction Pipeline", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Visualization saved to {save_path}")


if __name__ == "__main__":
    visualize_pipeline(r"verifnet/data/raw/p1/p1.bmp")
