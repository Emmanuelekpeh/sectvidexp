"""Camera motion estimation from background features (spec 6.2).

Returns a 2x3 affine A (in pixel coordinates) such that p_cur = A p_prev
for background points, plus a 0..1 confidence. Translation A[:2, 2] is in px.
"""
import cv2
import numpy as np

_IDENT = np.eye(2, 3, dtype=np.float32)


def background_mask(person_mask, margin_px=8):
    """Background = everything except a band of `margin_px` around the person."""
    bg = (person_mask == 0).astype(np.uint8) * 255
    if margin_px > 0:
        k = 2 * margin_px + 1
        bg = cv2.erode(bg, np.ones((k, k), np.uint8))
    return bg


def estimate_background_affine(prev_gray, prev_bg_mask, cur_gray, cur_bg_mask,
                               max_features=500, ransac_thresh_px=3.0, min_inliers=30):
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, d1 = orb.detectAndCompute(cv2.GaussianBlur(prev_gray, (3, 3), 0), mask=prev_bg_mask)
    kp2, d2 = orb.detectAndCompute(cv2.GaussianBlur(cur_gray, (3, 3), 0), mask=cur_bg_mask)
    if d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8:
        return _IDENT.copy(), 0.0

    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = []
    if matches:
        for pair in matches:
            best, second = pair[0], pair[1]
            if best.distance < 0.75 * second.distance:
                good.append(best)
    if len(good) < min_inliers:
        return _IDENT.copy(), 0.0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    A, inliers = cv2.estimateAffine2D(
        pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh_px
    )
    if A is None:
        return _IDENT.copy(), 0.0
    conf = float(inliers.sum()) / max(len(good), 1)
    return A.astype(np.float32), conf
