import numpy as np
import torch.nn.functional as F


######################################################################
# The function to get the error in percentage of inter ocular is from:
# https://github.com/jamt9000/DVE
######################################################################


def inter_ocular_error(output, kp_normalized, eyeidxs):
    pred = output.detach().cpu()
    gt = kp_normalized.detach().cpu()
    iod = ((gt[:, eyeidxs[0], :] - gt[:, eyeidxs[1], :])**2.).sum(1).sqrt()[:, None]
    err = ((pred - gt)**2.).sum(2).sqrt()
    ioderr = err / iod
    return 100 * ioderr.mean()


######################################################################
# The following functions to compute the PCK metric are from:
# https://github.com/guopei/PoseEstimation-FCN-Pytorch
######################################################################


def get_dists(preds, gts):
    (batches, channels) = preds.shape[:2]
    dists = np.zeros((channels, batches), np.int32)
    for b in range(batches):
        for c in range(channels):
            if gts[b, c, 0] > 0 and gts[b, c, 1] > 0:
                dists[c,b] = ((gts[b,c] - preds[b,c]) ** 2).sum() ** 0.5
            else:
                dists[c,b] = -1
    return dists


def within_threshold(dist, outsize, thr = 0.05):
    dist = dist[dist != -1]
    if len(dist) > 0:
        return (dist < thr * outsize).sum() / float(len(dist))
    else:
        return -1

def kp_unnormalize(H, W, kp):
    kp = kp.copy()
    kp[..., 0] = (kp[..., 0] + 1)  * (W - 1) / 2
    kp[..., 1] = (kp[..., 1] + 1)  * (H - 1) / 2
    return kp

def calc_pck(output, target, visible, boxsize):
    preds = output.detach().cpu().numpy()
    gts   = target.detach().cpu().numpy() # normalized kp
    B, nparts, _ = gts.shape
    
    preds = kp_unnormalize(boxsize, boxsize, preds)
    gts   = kp_unnormalize(boxsize, boxsize, gts)

    for i in range(B):
        vis = visible[i]
        invis = [not i for i in vis]
        gts[i][invis] = 0 

    dists = get_dists(preds, gts)
    acc = np.zeros(nparts, dtype=np.float32)
    avg_ccc = 0.0
    bad_idx_count = 0

    for i in range(nparts):
        acc[i] = within_threshold(dists[i], boxsize)
        if acc[i] >= 0:
            avg_ccc = avg_ccc + acc[i]
        else:
            bad_idx_count = bad_idx_count + 1
  
    if bad_idx_count == nparts:
        return 0
    else:
        return avg_ccc / (nparts - bad_idx_count) * 100


