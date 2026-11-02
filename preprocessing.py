"""
Preprocessing pipeline for VerifNet contactless fingerprint recognition.
Implements:
1. K-Means clustering for finger segmentation and background removal.
2. Cropping to region of interest (ROI).
3. Adaptive Gaussian Thresholding to handle uneven illumination.
4. Gabor filter enhancement to accentuate ridge-valley textures.
5. Thinning / Skeletonization to 1-pixel wide ridges.
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize


class FingerprintPreprocessor:
    def __init__(self, target_size=(128, 128), gabor_orientations=8, ksize=15, sigma=3.0, lambd=8.0):
        self.target_size = target_size
        self.gabor_orientations = gabor_orientations
        self.ksize = ksize
        self.sigma = sigma
        self.lambd = lambd
        self.gabor_kernels = self._build_gabor_bank()

    def _build_gabor_bank(self):
        """Construct a multi-orientation bank of Gabor filters."""
        kernels = []
        for i in range(self.gabor_orientations):
            theta = i * np.pi / self.gabor_orientations
            kernel = cv2.getGaborKernel(
                (self.ksize, self.ksize),
                sigma=self.sigma,
                theta=theta,
                lambd=self.lambd,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F
            )
            kernel /= 1.5 * np.maximum(kernel.sum(), 1e-5)
            kernels.append(kernel)
        return kernels

    def segment_kmeans(self, gray_img):
        """
        Segment finger from background using K-Means clustering (k=2).
        Returns binary mask (1 for finger, 0 for background).
        """
        h, w = gray_img.shape
        # Downsample for rapid cluster center estimation
        small = cv2.resize(gray_img, (140, 90))
        pixels = small.reshape((-1, 1)).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)

        # In contactless fingerphotos against neutral/dark background, finger is the brighter cluster
        finger_idx = np.argmax(centers)
        mask_small = (labels == finger_idx).astype(np.uint8).reshape(small.shape)

        # Upsample mask to original resolution
        mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

        # Morphological smoothing: close small holes, open isolated noise
        morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_k)
        return mask

    def process(self, img_bgr_or_gray):
        """
        Full preprocessing pipeline.
        Returns dictionary containing:
        - 'mask': binary finger mask
        - 'enhanced': Gabor-enhanced normalized grayscale image [0, 1]
        - 'adaptive_thresh': binary ridge image [0, 255]
        - 'skeleton': 1-pixel wide skeletonized binary image [0, 1]
        - 'dist_map': distance transform from finger boundary (for minutiae filtering)
        """
        if len(img_bgr_or_gray.shape) == 3:
            gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr_or_gray.copy()

        # Step 1: K-Means Segmentation
        mask = self.segment_kmeans(gray)
        y_idx, x_idx = np.where(mask > 0)
        if len(y_idx) == 0:
            # Fallback if segmentation fails
            ymin, ymax, xmin, xmax = 0, gray.shape[0], 0, gray.shape[1]
            mask = np.ones_like(gray, dtype=np.uint8)
        else:
            ymin, ymax = y_idx.min(), y_idx.max()
            xmin, xmax = x_idx.min(), x_idx.max()

        cropped_gray = (gray * mask)[ymin:ymax, xmin:xmax]
        cropped_mask = mask[ymin:ymax, xmin:xmax]

        # Resize to standard input size (e.g., 128x128 for Scattering2D / CNN)
        roi_gray = cv2.resize(cropped_gray, self.target_size, interpolation=cv2.INTER_AREA)
        roi_mask = cv2.resize(cropped_mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # Step 2: Adaptive Gaussian Thresholding
        # Handles non-uniform illumination across the curved finger
        adaptive_thresh = cv2.adaptiveThreshold(
            roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 2
        )
        adaptive_thresh = adaptive_thresh * (roi_mask > 0)

        # Step 3: Gabor Filter Enhancement
        accum = np.zeros_like(roi_gray, dtype=np.float32)
        for kern in self.gabor_kernels:
            fimg = cv2.filter2D(roi_gray, cv2.CV_32F, kern)
            np.maximum(accum, fimg, out=accum)
        enhanced = np.clip(accum, 0, 255).astype(np.uint8) * (roi_mask > 0)

        # Step 4: Thinning / Skeletonization
        binary_ridges = (adaptive_thresh > 128)
        skeleton = skeletonize(binary_ridges).astype(np.uint8) * (roi_mask > 0)

        # Step 5: Distance transform from mask boundary (for minutiae filtering)
        dist_map = cv2.distanceTransform(roi_mask.astype(np.uint8), cv2.DIST_L2, 5)

        # Normalized enhanced image for deep learning (in range [0, 1])
        enhanced_float = enhanced.astype(np.float32) / 255.0

        return {
            'roi_gray': roi_gray,
            'mask': roi_mask,
            'adaptive_thresh': adaptive_thresh,
            'enhanced': enhanced_float,
            'enhanced_uint8': enhanced,
            'skeleton': skeleton,
            'dist_map': dist_map
        }
