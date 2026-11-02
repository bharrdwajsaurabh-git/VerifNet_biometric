"""
End-to-End Training and Evaluation Pipeline for VerifNet.
Implements:
1. Subject-level split (Train vs Unseen Test subjects).
2. Contrastive training of the Wavelet-Siamese network.
3. Minutiae matching on test pairs.
4. Score Fusion (0.90 * DL + 0.10 * Minutiae).
5. Comprehensive Biometric Evaluation: EER, AUROC, AUPR, FMR1000, FMR10000.
6. Fusion weight ablation (matching Table II in the paper).
7. ROC & DET curve generation.
"""

import os
import random
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from verifnet.dataset import FingerprintDataset, PairDataset
from verifnet.wavelet_siamese import VerifNetSiamese, ContrastiveLoss
from verifnet.minutiae_matcher import MinutiaeMatcher
from verifnet.fusion_evaluator import FusionEvaluator


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_pipeline(data_dir="verifnet/data/raw",
                 cache_dir="verifnet/data/cache",
                 epochs=15,
                 batch_size=32,
                 train_pairs_per_epoch=800,
                 test_pairs_count=200):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing VerifNet pipeline on device: {device}", flush=True)

    # Discover subjects
    subjs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if len(subjs) < 10:
        raise ValueError(f"Need at least 10 subjects in {data_dir}, found {len(subjs)}")

    # Split: 70% train subjects, 30% test subjects (strict unseen evaluation)
    split_idx = int(0.70 * len(subjs))
    train_subjs = subjs[:split_idx]
    test_subjs = subjs[split_idx:]
    print(f"Subjects: {len(train_subjs)} Training, {len(test_subjs)} Testing (unseen).", flush=True)

    # Initialize Siamese Model and Matcher
    print("\n--- Step 1: Initializing VerifNet Model ---", flush=True)
    model = VerifNetSiamese(img_size=(128, 128), J=2, L=8, embedding_dim=16).to(device)
    loss_fn = ContrastiveLoss(margin=1.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    matcher = MinutiaeMatcher()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VerifNet Siamese CNN initialized with {total_params:,} trainable parameters.", flush=True)

    # Load & Precompute Datasets
    print("\n--- Step 2: Loading & Precomputing Datasets ---", flush=True)
    train_dataset = FingerprintDataset(data_dir, matcher=matcher, siamese_model=model, cache_dir=cache_dir, subjects=train_subjs)
    test_dataset = FingerprintDataset(data_dir, matcher=matcher, siamese_model=model, cache_dir=cache_dir, subjects=test_subjs)

    # Training Loop
    print("\n--- Step 3: Training Wavelet-Siamese Network ---", flush=True)
    start_train_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_pairs = train_dataset.create_pairs(num_pairs=train_pairs_per_epoch, genuine_ratio=0.5)
        pair_loader = DataLoader(PairDataset(train_dataset, train_pairs), batch_size=batch_size, shuffle=True)

        epoch_loss = 0.0
        total_batches = 0
        for s1, s2, targets, _, _ in pair_loader:
            s1 = s1.to(device)
            s2 = s2.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            _, _, dw = model(s1, s2)
            loss = loss_fn(dw, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total_batches += 1

        avg_loss = epoch_loss / max(total_batches, 1)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Contrastive Loss: {avg_loss:.4f}", flush=True)

    train_duration = time.time() - start_train_time
    print(f"Training completed in {train_duration:.2f} seconds.", flush=True)

    # Save model checkpoint
    os.makedirs("verifnet/checkpoints", exist_ok=True)
    model_save_path = "verifnet/checkpoints/verifnet_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}", flush=True)

    # Step 4: Testing & Score Fusion
    print("\n--- Step 4: Testing on Unseen Subject Pairs ---", flush=True)
    test_pairs = test_dataset.create_pairs(num_pairs=test_pairs_count, genuine_ratio=0.5)
    test_loader = DataLoader(PairDataset(test_dataset, test_pairs), batch_size=batch_size, shuffle=False)

    model.eval()
    y_true = []
    dl_scores = []
    minutiae_scores = []

    with torch.no_grad():
        for s1, s2, targets, p1_list, p2_list in test_loader:
            s1 = s1.to(device)
            s2 = s2.to(device)

            _, _, dw = model(s1, s2)
            sim = torch.clamp(1.0 - (dw / 1.5), min=0.0, max=1.0)
            dl_scores.extend(sim.cpu().numpy().tolist())
            y_true.extend(targets.numpy().astype(int).tolist())

            for p1, p2 in zip(p1_list, p2_list):
                m1 = test_dataset.get_sample(p1)['minutiae']
                m2 = test_dataset.get_sample(p2)['minutiae']
                m_score = matcher.match(m1, m2)
                minutiae_scores.append(m_score)

    y_true = np.array(y_true)
    dl_scores = np.array(dl_scores)
    minutiae_scores = np.array(minutiae_scores)

    evaluator = FusionEvaluator(dl_weight=0.90, minutiae_weight=0.10)
    fused_scores = evaluator.fuse_scores(dl_scores, minutiae_scores)

    # Compute individual & fused metrics
    res_dl = evaluator.compute_metrics(y_true, dl_scores)
    res_mm = evaluator.compute_metrics(y_true, minutiae_scores)
    res_fused = evaluator.compute_metrics(y_true, fused_scores)

    print("\n==================================================================", flush=True)
    print("                    VERIFNET BENCHMARK RESULTS                    ", flush=True)
    print("==================================================================", flush=True)
    print(f"{'Method':<25} | {'EER (%)':<10} | {'AUROC':<8} | {'AUPR':<8} | {'FMR1000':<8}", flush=True)
    print("------------------------------------------------------------------", flush=True)
    print(f"{'Minutiae Matching (MM)':<25} | {res_mm['eer']*100:>8.2f}% | {res_mm['auroc']:>8.4f} | {res_mm['aupr']:>8.4f} | {res_mm['fmr1000']:>8.4f}", flush=True)
    print(f"{'Deep Learning (SWT+CNN)':<25} | {res_dl['eer']*100:>8.2f}% | {res_dl['auroc']:>8.4f} | {res_dl['aupr']:>8.4f} | {res_dl['fmr1000']:>8.4f}", flush=True)
    print(f"{'VerifNet (Fused 90/10)':<25} | {res_fused['eer']*100:>8.2f}% | {res_fused['auroc']:>8.4f} | {res_fused['aupr']:>8.4f} | {res_fused['fmr1000']:>8.4f}", flush=True)
    print("==================================================================", flush=True)

    # Step 5: Fusion Weight Ablation (Matching Paper Table II)
    print("\n--- Step 5: Score Fusion Weight Ablation Study (Paper Table II) ---", flush=True)
    print(f"{'DL Weight':<12} | {'MM Weight':<12} | {'AUROC':<10} | {'AUPR':<10} | {'EER (%)':<10}", flush=True)
    print("------------------------------------------------------------------", flush=True)
    weights = [
        (0.10, 0.90),
        (0.20, 0.80),
        (0.30, 0.70),
        (0.40, 0.60),
        (0.50, 0.50),
        (0.60, 0.40),
        (0.70, 0.30),
        (0.80, 0.20),
        (0.90, 0.10),
        (1.00, 0.00)
    ]
    for w_dl, w_mm in weights:
        fused = w_dl * dl_scores + w_mm * minutiae_scores
        metrics = evaluator.compute_metrics(y_true, fused)
        is_best = " (Proposed)" if (w_dl == 0.90 and w_mm == 0.10) else ""
        print(f"{w_dl*100:>4.0f}% DL       | {w_mm*100:>4.0f}% MM       | {metrics['auroc']:>8.4f}   | {metrics['aupr']:>8.4f}   | {metrics['eer']*100:>7.2f}%{is_best}", flush=True)

    # Step 6: Plot ROC & DET Curves
    print("\n--- Step 6: Generating ROC and DET Curves ---", flush=True)
    results_map = {
        'Minutiae Matching': res_mm,
        'Deep Learning (SWT+CNN)': res_dl,
        'VerifNet Fused (90% DL + 10% MM)': res_fused
    }
    curve_path = "verifnet/verifnet_curves.png"
    evaluator.plot_roc_det(results_map, save_path=curve_path)

    return {
        'res_dl': res_dl,
        'res_mm': res_mm,
        'res_fused': res_fused
    }


if __name__ == "__main__":
    run_pipeline()
