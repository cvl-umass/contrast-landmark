import torch.nn.functional as F
import time
import torch


def regression_loss(prediction_normalized, kp_normalized, alpha=10., **kwargs):
    kp = kp_normalized.to(prediction_normalized.device)
    B, nA, _ = prediction_normalized.shape
    return F.smooth_l1_loss(prediction_normalized * alpha, kp * alpha)


def selected_regression_loss(prediction_normalized, kp_normalized, visible, alpha=10., **kwargs):
    kp = kp_normalized.to(prediction_normalized.device)
    B, nA, _ = prediction_normalized.shape
    for i in range(B):
        vis = visible[i]
        invis = [not v for v in vis]
        kp[i][invis] = 0.
        prediction_normalized[i][invis] = 0.

    return F.smooth_l1_loss(prediction_normalized * alpha, kp * alpha)
