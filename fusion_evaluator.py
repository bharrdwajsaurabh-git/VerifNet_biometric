"""
Score-level fusion and biometric evaluation metrics for VerifNet.
Implements:
1. Score fusion: S_fused = 0.90 * S_DL + 0.10 * S_minutiae.
2. Equal Error Rate (EER) calculation.
3. False Match Rate (FMR) and False Non-Match Rate (FNMR) curves.
4. FMR1000 and FMR10000 metrics.
5. AUROC and AUPR calculation.
6. Plotting ROC and DET curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


class FusionEvaluator:
    def __init__(self, dl_weight=0.90, minutiae_weight=0.10):
        self.dl_weight = dl_weight
        self.minutiae_weight = minutiae_weight

    def fuse_scores(self, dl_scores, minutiae_scores):
        """
        Computes weighted score fusion:
        S = w_dl * S_dl + w_minutiae * S_minutiae
        """
        dl_scores = np.asarray(dl_scores, dtype=np.float32)
        minutiae_scores = np.asarray(minutiae_scores, dtype=np.float32)
        return self.dl_weight * dl_scores + self.minutiae_weight * minutiae_scores

    def compute_metrics(self, y_true, scores):
        """
        Computes comprehensive biometric verification metrics:
        - EER (Equal Error Rate)
        - Threshold at EER
        - AUROC
        - AUPR
        - FMR1000 (FNMR at FMR = 0.1%)
        - FMR10000 (FNMR at FMR = 0.01%)
        """
        y_true = np.asarray(y_true, dtype=np.int32)
        scores = np.asarray(scores, dtype=np.float32)

        fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
        roc_auc = auc(fpr, tpr)

        # FMR = FPR, FNMR = 1 - TPR
        fmr = fpr
        fnmr = 1.0 - tpr

        # Equal Error Rate (EER): where FMR == FNMR
        eer_idx = np.nanargmin(np.abs(fmr - fnmr))
        eer = (fmr[eer_idx] + fnmr[eer_idx]) / 2.0
        eer_threshold = thresholds[eer_idx]

        # Precision-Recall
        precision, recall, _ = precision_recall_curve(y_true, scores)
        aupr = auc(recall, precision)

        # FMR1000 (FNMR when FMR <= 0.001)
        idx_1000 = np.where(fmr <= 0.001)[0]
        fmr1000 = fnmr[idx_1000[-1]] if len(idx_1000) > 0 else fnmr[0]

        # FMR10000 (FNMR when FMR <= 0.0001)
        idx_10000 = np.where(fmr <= 0.0001)[0]
        fmr10000 = fnmr[idx_10000[-1]] if len(idx_10000) > 0 else fnmr[0]

        return {
            'eer': float(eer),
            'eer_threshold': float(eer_threshold),
            'auroc': float(roc_auc),
            'aupr': float(aupr),
            'fmr1000': float(fmr1000),
            'fmr10000': float(fmr10000),
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'fmr': fmr,
            'fnmr': fnmr
        }

    def plot_roc_det(self, results_dict, save_path="verifnet_curves.png"):
        """
        Generates comparison plots of ROC and DET curves for:
        - Deep Learning alone
        - Minutiae Matching alone
        - VerifNet Fused
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 1. ROC Curves
        ax_roc = axes[0]
        for name, res in results_dict.items():
            ax_roc.plot(res['fpr'], res['tpr'], lw=2,
                        label=f"{name} (AUC={res['auroc']:.4f}, EER={res['eer']*100:.2f}%)")
        ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate (FMR)', fontsize=12)
        ax_roc.set_ylabel('True Positive Rate (1 - FNMR)', fontsize=12)
        ax_roc.set_title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
        ax_roc.legend(loc='lower right', fontsize=10)
        ax_roc.grid(True, linestyle=':', alpha=0.6)

        # 2. DET Curves (FNMR vs FMR)
        ax_det = axes[1]
        for name, res in results_dict.items():
            ax_det.plot(res['fmr'] * 100, res['fnmr'] * 100, lw=2, label=name)
        ax_det.set_xlabel('False Match Rate - FMR (%)', fontsize=12)
        ax_det.set_ylabel('False Non-Match Rate - FNMR (%)', fontsize=12)
        ax_det.set_title('Detection Error Tradeoff (DET)', fontsize=14, fontweight='bold')
        ax_det.legend(loc='upper right', fontsize=10)
        ax_det.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"ROC and DET curves saved to {save_path}")
