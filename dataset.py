"""
Dataset and Pair Generator for VerifNet Contactless Fingerprint Verification.
Handles loading raw images, running preprocessing, caching enhanced images,
wavelet scattering coefficients, and minutiae points, and generating balanced
genuine/imposter pairs for ultra-fast Siamese training and evaluation.
"""

import os
import glob
import random
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from verifnet.preprocessing import FingerprintPreprocessor
from verifnet.minutiae_matcher import MinutiaeMatcher
from verifnet.wavelet_siamese import VerifNetSiamese


class FingerprintDataset(Dataset):
    def __init__(self, data_dir, preprocessor=None, matcher=None, siamese_model=None, cache_dir=None, subjects=None):
        self.data_dir = data_dir
        self.preprocessor = preprocessor or FingerprintPreprocessor()
        self.matcher = matcher or MinutiaeMatcher()
        self.siamese_model = siamese_model or VerifNetSiamese(img_size=(128, 128))
        self.cache_dir = cache_dir

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        all_subjs = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        if subjects is not None:
            self.subjects = [s for s in all_subjs if s in subjects]
        else:
            self.subjects = all_subjs

        self.subject_images = {}
        for s in self.subjects:
            s_dir = os.path.join(data_dir, s)
            imgs = sorted(glob.glob(os.path.join(s_dir, "*.bmp")) +
                          glob.glob(os.path.join(s_dir, "*.jpg")) +
                          glob.glob(os.path.join(s_dir, "*.png")))
            if len(imgs) > 0:
                self.subject_images[s] = imgs

        self.valid_subjects = [s for s, imgs in self.subject_images.items() if len(imgs) >= 2]
        print(f"FingerprintDataset: {len(self.valid_subjects)} valid subjects, {sum(len(v) for v in self.subject_images.values())} total images.", flush=True)

        self.cache = {}
        self._precompute_all()

    def _precompute_all(self):
        """Precomputes and caches enhanced images, scattering features, and minutiae points."""
        print("Precomputing enhanced images, scattering features, and minutiae points...", flush=True)
        count = 0
        for s in self.valid_subjects:
            for img_path in self.subject_images[s]:
                cache_key = os.path.basename(s) + "_" + os.path.basename(img_path)
                disk_file = os.path.join(self.cache_dir, cache_key + ".npz") if self.cache_dir else None

                if disk_file and os.path.exists(disk_file):
                    data = np.load(disk_file)
                    self.cache[img_path] = {
                        'enhanced': data['enhanced'],
                        'scat': data['scat'] if 'scat' in data else None,
                        'minutiae': data['minutiae'],
                        'subject': s
                    }
                    if self.cache[img_path]['scat'] is None:
                        t = torch.from_numpy(data['enhanced']).unsqueeze(0).unsqueeze(0).float()
                        with torch.no_grad():
                            scat_feat = self.siamese_model.extract_scattering(t).squeeze(0).numpy()
                        self.cache[img_path]['scat'] = scat_feat
                        np.savez_compressed(disk_file, enhanced=data['enhanced'], minutiae=data['minutiae'], scat=scat_feat)
                else:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    res = self.preprocessor.process(img)
                    minutiae = self.matcher.extract_minutiae(res['skeleton'], res['mask'], res['dist_map'])
                    
                    t = torch.from_numpy(res['enhanced']).unsqueeze(0).unsqueeze(0).float()
                    with torch.no_grad():
                        scat_feat = self.siamese_model.extract_scattering(t).squeeze(0).numpy()

                    self.cache[img_path] = {
                        'enhanced': res['enhanced'],
                        'scat': scat_feat,
                        'minutiae': minutiae,
                        'subject': s
                    }
                    if disk_file:
                        np.savez_compressed(disk_file, enhanced=res['enhanced'], minutiae=minutiae, scat=scat_feat)
                count += 1
                if count % 50 == 0:
                    print(f"  Processed {count} images...", flush=True)
        print(f"Precomputing complete: {count} images cached and ready.", flush=True)

    def get_sample(self, img_path):
        return self.cache[img_path]

    def create_pairs(self, num_pairs=1000, genuine_ratio=0.5):
        pairs = []
        num_genuine = int(num_pairs * genuine_ratio)
        num_imposter = num_pairs - num_genuine

        # Genuine pairs
        for _ in range(num_genuine):
            s = random.choice(self.valid_subjects)
            imgs = self.subject_images[s]
            img1, img2 = random.sample(imgs, 2)
            pairs.append((img1, img2, 1.0))

        # Imposter pairs
        for _ in range(num_imposter):
            s1, s2 = random.sample(self.valid_subjects, 2)
            img1 = random.choice(self.subject_images[s1])
            img2 = random.choice(self.subject_images[s2])
            pairs.append((img1, img2, 0.0))

        random.shuffle(pairs)
        return pairs


class PairDataset(Dataset):
    def __init__(self, fingerprint_dataset, pairs):
        self.fp_dataset = fingerprint_dataset
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path1, img_path2, label = self.pairs[idx]
        sample1 = self.fp_dataset.get_sample(img_path1)
        sample2 = self.fp_dataset.get_sample(img_path2)

        s1 = torch.from_numpy(sample1['scat']).float()
        s2 = torch.from_numpy(sample2['scat']).float()
        target = torch.tensor(label, dtype=torch.float32)

        return s1, s2, target, img_path1, img_path2
