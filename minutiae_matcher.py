"""
Minutiae Extraction and Matching module for VerifNet.
Implements:
1. Fast Crossing Number (CN) detection for ridge terminations (CN=1) and bifurcations (CN=3).
2. Orientation angle theta computation.
3. Boundary false minutiae elimination using distance-to-mask transform.
4. Minutiae spatial alignment transform (Eq 4).
5. Distance and angle-based matching and score calculation S = n^2 / (N1 * N2) (Eq 5).
"""

import numpy as np
import scipy.signal


class MinutiaeMatcher:
    def __init__(self, border_margin=10, min_dist=5, match_dist_thresh=12, match_angle_thresh=np.pi / 4):
        self.border_margin = border_margin
        self.min_dist = min_dist
        self.match_dist_thresh = match_dist_thresh
        self.match_angle_thresh = match_angle_thresh

    def extract_minutiae(self, skeleton, mask, dist_map=None):
        """
        Extracts minutiae points: [x, y, theta, CN]
        CN = 1: termination
        CN = 3: bifurcation
        """
        if dist_map is None:
            import cv2
            dist_map = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)

        skel = (skeleton > 0).astype(np.uint8)
        H, W = skel.shape

        # Quick neighbor count using 2D convolution
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)
        neighbor_count = scipy.signal.convolve2d(skel, kernel, mode='same')

        # Minutiae only occur where skel==1 and sufficiently far from mask edge
        valid_mask = (skel == 1) & (dist_map > self.border_margin)
        candidate_y, candidate_x = np.where(valid_mask & ((neighbor_count == 1) | (neighbor_count == 3)))

        minutiae = []
        pad_skel = np.pad(skel.astype(np.int32), 1, mode='constant', constant_values=0)

        for y, x in zip(candidate_y, candidate_x):
            if y < 1 or y >= H - 1 or x < 1 or x >= W - 1:
                continue

            py, px = y + 1, x + 1
            # 8 neighbors in clockwise order:
            # (py-1, px), (py-1, px+1), (py, px+1), (py+1, px+1),
            # (py+1, px), (py+1, px-1), (py, px-1), (py-1, px-1)
            p = [
                int(pad_skel[py-1, px]),
                int(pad_skel[py-1, px+1]),
                int(pad_skel[py,   px+1]),
                int(pad_skel[py+1, px+1]),
                int(pad_skel[py+1, px]),
                int(pad_skel[py+1, px-1]),
                int(pad_skel[py,   px-1]),
                int(pad_skel[py-1, px-1])
            ]

            # Crossing Number: CN = 0.5 * sum(|p_i - p_{i+1}|)
            cn = 0.5 * sum(abs(p[i] - p[(i + 1) % 8]) for i in range(8))

            if cn == 1.0 or cn == 3.0:
                active_idx = [i for i, v in enumerate(p) if v == 1]
                angles = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4]
                theta = angles[active_idx[0]] if active_idx else 0.0
                minutiae.append([float(x), float(y), float(theta), float(cn)])

        if len(minutiae) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        m_arr = np.array(minutiae, dtype=np.float32)

        # Minutiae suppression (filter dense spurious clusters within min_dist)
        keep = []
        for i in range(len(m_arr)):
            too_close = False
            for k in keep:
                if np.hypot(m_arr[i, 0] - m_arr[k, 0], m_arr[i, 1] - m_arr[k, 1]) < self.min_dist:
                    too_close = True
                    break
            if not too_close:
                keep.append(i)

        return m_arr[keep]

    def match(self, M1, M2, max_candidates=25):
        """
        Calculates similarity score S between minutiae sets M1 and M2.
        S = n^2 / (N1 * N2) (Eq 5 in the paper)
        Returns float similarity score in [0.0, 1.0].
        """
        if len(M1) < 5 or len(M2) < 5:
            return 0.0

        N1, N2 = len(M1), len(M2)
        max_score = 0.0

        # Choose candidate reference points in M1
        cand_indices_1 = np.random.choice(N1, min(N1, max_candidates), replace=False)

        for i in cand_indices_1:
            x1_ref, y1_ref, th1_ref, cn1 = M1[i]

            # Transform M1 relative to reference minutia M1[i] (Eq 4)
            dx1 = M1[:, 0] - x1_ref
            dy1 = M1[:, 1] - y1_ref
            cos1, sin1 = np.cos(th1_ref), np.sin(th1_ref)
            T1_x = cos1 * dx1 + sin1 * dy1
            T1_y = -sin1 * dx1 + cos1 * dy1
            T1_th = (M1[:, 2] - th1_ref + np.pi) % (2 * np.pi) - np.pi
            T1_cn = M1[:, 3]

            # Find matching candidate reference points in M2 having the same CN
            same_cn_idx = np.where(M2[:, 3] == cn1)[0]
            if len(same_cn_idx) == 0:
                continue

            cand_indices_2 = np.random.choice(same_cn_idx, min(len(same_cn_idx), max_candidates), replace=False)

            for j in cand_indices_2:
                x2_ref, y2_ref, th2_ref, _ = M2[j]

                # Transform M2 relative to reference minutia M2[j]
                dx2 = M2[:, 0] - x2_ref
                dy2 = M2[:, 1] - y2_ref
                cos2, sin2 = np.cos(th2_ref), np.sin(th2_ref)
                T2_x = cos2 * dx2 + sin2 * dy2
                T2_y = -sin2 * dx2 + cos2 * dy2
                T2_th = (M2[:, 2] - th2_ref + np.pi) % (2 * np.pi) - np.pi

                # Pairwise distances and angle differences
                # Shapes: (N1, N2)
                dists = np.sqrt((T1_x[:, None] - T2_x[None, :]) ** 2 + (T1_y[:, None] - T2_y[None, :]) ** 2)
                th_diff = np.abs((T1_th[:, None] - T2_th[None, :] + np.pi) % (2 * np.pi) - np.pi)
                cn_match = (T1_cn[:, None] == M2[None, :, 3])

                valid_pairs = (dists < self.match_dist_thresh) & (th_diff < self.match_angle_thresh) & cn_match

                # Count number of matched points
                n_matched = np.sum(np.any(valid_pairs, axis=1))
                score = float((n_matched ** 2) / (N1 * N2))

                if score > max_score:
                    max_score = score
                    if max_score >= 1.0:
                        return 1.0

        return min(max_score, 1.0)
