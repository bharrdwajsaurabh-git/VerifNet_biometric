"""
Inference & Verification Demo for VerifNet.
Takes two fingerprint images as input and performs contactless verification:
1. Preprocesses both fingerphotos (K-Means, Gabor enhancement, Thinning).
2. Extracts Minutiae points & computes Minutiae match score (S_MM).
3. Computes Scattering Wavelet Siamese deep learning similarity score (S_DL).
4. Fuses scores: S_fused = 0.90 * S_DL + 0.10 * S_MM.
5. Outputs verification verdict based on calibrated threshold.
"""

import argparse
import os
import sys

# Ensure current directory is on python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch

from verifnet.preprocessing import FingerprintPreprocessor
from verifnet.minutiae_matcher import MinutiaeMatcher
from verifnet.wavelet_siamese import VerifNetSiamese


def verify_pair(img_path1, img_path2, model_path="verifnet/checkpoints/verifnet_model.pth", threshold=0.50):
    if not os.path.exists(img_path1):
        raise FileNotFoundError(f"Image not found: {img_path1}")
    if not os.path.exists(img_path2):
        raise FileNotFoundError(f"Image not found: {img_path2}")

    print("==================================================================")
    print("                VERIFNET CONTACTLESS VERIFICATION                 ")
    print("==================================================================")
    print(f"Sample 1: {img_path1}")
    print(f"Sample 2: {img_path2}")

    # Step 1: Preprocess images
    print("\n[Step 1] Running Adaptive Preprocessing & Enhancement...")
    prep = FingerprintPreprocessor(target_size=(128, 128))
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    res1 = prep.process(img1)
    res2 = prep.process(img2)

    # Step 2: Minutiae Extraction and Matching
    print("[Step 2] Extracting Minutiae & Performing Spatial Alignment...")
    matcher = MinutiaeMatcher()
    m1 = matcher.extract_minutiae(res1['skeleton'], res1['mask'], res1['dist_map'])
    m2 = matcher.extract_minutiae(res2['skeleton'], res2['mask'], res2['dist_map'])
    minutiae_score = matcher.match(m1, m2)
    print(f"  -> Extracted {len(m1)} minutiae from Sample 1, {len(m2)} from Sample 2.")
    print(f"  -> Classical Minutiae Match Score (S_MM): {minutiae_score:.4f}")

    # Step 3: Deep Learning Wavelet-Siamese Inference
    print("[Step 3] Extracting Wavelet Scattering Features & CNN Embeddings...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VerifNetSiamese(img_size=(128, 128), J=2, L=8, embedding_dim=16).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"  -> Loaded trained weights from {model_path}")
    else:
        print(f"  -> Warning: checkpoint {model_path} not found, running evaluation mode.")

    model.eval()
    t1 = torch.from_numpy(res1['enhanced']).unsqueeze(0).unsqueeze(0).float().to(device)
    t2 = torch.from_numpy(res2['enhanced']).unsqueeze(0).unsqueeze(0).float().to(device)

    dl_score = float(model.compute_similarity(t1, t2)[0])
    print(f"  -> Deep Learning Siamese Similarity Score (S_DL): {dl_score:.4f}")

    # Step 4: Score-Level Fusion
    print("\n[Step 4] Performing Score-Level Fusion (90% DL + 10% Minutiae)...")
    fused_score = 0.90 * dl_score + 0.10 * minutiae_score
    print(f"  -> VerifNet Fused Score: {fused_score:.4f} (Decision Threshold: {threshold:.4f})")

    # Step 5: Verification Decision
    is_match = fused_score >= threshold
    verdict = "MATCH (AUTHENTICATED - SAME PERSON)" if is_match else "NON-MATCH (REJECTED - DIFFERENT PERSON)"

    print("\n==================================================================")
    print(f"VERDICT: {verdict}")
    print("==================================================================")

    return {
        'img1': img_path1,
        'img2': img_path2,
        'minutiae_score': minutiae_score,
        'dl_score': dl_score,
        'fused_score': fused_score,
        'is_match': is_match
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VerifNet Contactless Verification CLI")
    parser.add_argument("--img1", type=str, default="verifnet/data/raw/p1/p1.bmp", help="Path to first fingerprint image")
    parser.add_argument("--img2", type=str, default="verifnet/data/raw/p1/p2.bmp", help="Path to second fingerprint image")
    parser.add_argument("--model", type=str, default="verifnet/checkpoints/verifnet_model.pth", help="Model weights path")
    parser.add_argument("--thresh", type=float, default=0.50, help="Verification threshold")
    args = parser.parse_args()

    verify_pair(args.img1, args.img2, model_path=args.model, threshold=args.thresh)
