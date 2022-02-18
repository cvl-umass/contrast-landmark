import torch
import torch.nn.functional as F
import math


def tps_grid(H, W):
    xi = torch.linspace(-1, 1, W)
    yi = torch.linspace(-1, 1, H)

    yy, xx = torch.meshgrid(yi, xi)
    grid = torch.stack((xx.reshape(-1), yy.reshape(-1)), 1)
    return grid


def spatial_grid_unnormalized(H, W):
    xi = torch.linspace(0, W - 1, W)
    yi = torch.linspace(0, H - 1, H)

    yy, xx = torch.meshgrid(yi, xi)
    grid = torch.stack((yy.reshape(-1), xx.reshape(-1)), 1) # modify the order of xx, yy in DVE
    return grid.reshape(H, W, 2)


def tps_U(grid1, grid2):
    D = grid1.reshape(-1, 1, 2) - grid2.reshape(1, -1, 2)
    D = torch.sum(D ** 2., 2)
    U = D * torch.log(D + 1e-5)
    return U


# def grid_unnormalize(grid, H, W):
# x = grid.reshape(-1, H, W, 2)
# x = (x + 1.) / 2. * torch.Tensor([W - 1., H - 1.]).reshape(1, 1, 1, 2).to(x.device)
# return x.reshape(grid.shape)

def grid_unnormalize(grid, H, W):
    x = grid.reshape(-1, H, W, 2)
    constants = torch.tensor([W - 1., H - 1.], dtype=x.dtype)
    constants = constants.reshape(1, 1, 1, 2).to(x.device)
    x = (x + 1.) / 2. * constants
    return x.reshape(grid.shape)


def grid_normalize(grid, H, W):
    x = grid.reshape(-1, H, W, 2)
    x = 2. * x / torch.Tensor([W - 1., H - 1.]).reshape(1, 1, 1, 2).to(x.device) - 1
    return x.reshape(grid.shape)


def random_tps_weights(nctrlpts, warpsd_all, warpsd_subset, transsd, scalesd, rotsd):
    W = torch.randn(nctrlpts, 2) * warpsd_all
    subset = torch.rand(W.shape) > 0.5
    W[subset] = torch.randn(subset.sum()) * warpsd_subset
    rot = torch.randn([]) * rotsd * math.pi / 180
    sc = 1. + torch.randn([]) * scalesd
    tx = torch.randn([]) * transsd
    ty = torch.randn([]) * transsd

    aff = torch.Tensor([[tx, ty],
                        [sc * torch.cos(rot), sc * -torch.sin(rot)],
                        [sc * torch.sin(rot), sc * torch.cos(rot)]])

    Wa = torch.cat((W, aff), 0)
    return Wa


class Warper(object):
    returns_pairs = True

    def __init__(self, H, W, warpsd_all=0.001, warpsd_subset=0.01, transsd=0.1,
                 scalesd=0.1, rotsd=5, im1_multiplier=0.5, im1_multiplier_aff=0.):
        self.H = H
        self.W = W
        self.warpsd_all = warpsd_all
        self.warpsd_subset = warpsd_subset
        self.transsd = transsd
        self.scalesd = scalesd
        self.rotsd = rotsd
        self.im1_multiplier = im1_multiplier
        self.im1_multiplier_aff = im1_multiplier_aff

        self.npixels = H * W
        self.nc = 10
        self.nctrlpts = self.nc * self.nc

        self.grid_pixels = tps_grid(H, W)
        self.grid_pixels_unnormalized = grid_unnormalize(self.grid_pixels.reshape(1, H, W, 2), self.H, self.W)
        self.grid_ctrlpts = tps_grid(self.nc, self.nc)
        self.U_ctrlpts = tps_U(self.grid_ctrlpts, self.grid_ctrlpts)
        self.U_pixels_ctrlpts = tps_U(self.grid_pixels, self.grid_ctrlpts)
        self.F = torch.cat((self.U_pixels_ctrlpts, torch.ones(self.npixels, 1), self.grid_pixels), 1)

    def __call__(self, im1, im2=None, keypts=None, crop=0):
        Hc = self.H - crop - crop
        Wc = self.W - crop - crop

        # im2 should be a copy of im1 with different colour jitter
        if im2 is None:
            im2 = im1

        kp1 = kp2 = 0

        unsqueezed = False
        if len(im1.shape) == 3:
            im1 = im1.unsqueeze(0)
            im2 = im2.unsqueeze(0)
            unsqueezed = True

        assert im1.shape[0] == 1 and im2.shape[0] == 1

        a = self.im1_multiplier
        b = self.im1_multiplier_aff
        weights1 = random_tps_weights(self.nctrlpts, a * self.warpsd_all, a * self.warpsd_subset, b * self.transsd,
                                      b * self.scalesd, b * self.rotsd)

        grid1 = torch.matmul(self.F, weights1).reshape(1, self.H, self.W, 2)
        grid1_unnormalized = grid_unnormalize(grid1, self.H, self.W) # [-1, 1] --> [H, W]
        if keypts is not None:
            kp1 = self.warp_keypoints(keypts, grid1_unnormalized)

        im1 = F.grid_sample(im1, grid1, align_corners = True)
        im2 = F.grid_sample(im2, grid1, align_corners = True)

        weights2 = random_tps_weights(self.nctrlpts, self.warpsd_all, self.warpsd_subset, self.transsd,
                                      self.scalesd, self.rotsd)
        grid2 = torch.matmul(self.F, weights2).reshape(1, self.H, self.W, 2)
        im2 = F.grid_sample(im2, grid2, align_corners = True)

        if crop != 0:
            im1 = im1[:, :, crop:-crop, crop:-crop]
            im2 = im2[:, :, crop:-crop, crop:-crop]

        if unsqueezed:
            im1 = im1.squeeze(0)
            im2 = im2.squeeze(0)

        grid = grid2
        grid_unnormalized = grid_unnormalize(grid, self.H, self.W)

        if keypts is not None:
            kp2 = self.warp_keypoints(kp1, grid_unnormalized)

        flow = grid_unnormalized - self.grid_pixels_unnormalized

        if crop != 0:
            flow = flow[:, crop:-crop, crop:-crop, :]

            grid_cropped = grid_unnormalized[:, crop:-crop, crop:-crop, :] - crop
            grid = grid_normalize(grid_cropped, Hc, Wc)

            # hc = flow.shape[1]
            # wc = flow.shape[2]
            # gridc = flow + grid_unnormalize(tps_grid(hc, wc).reshape(1, hc, wc, 2), hc, wc)
            # grid = grid_normalize(gridc, hc, wc)

            if keypts is not None:
                kp1 -= crop
                kp2 -= crop

        # Reverse the order because due to inverse warping the "flow" is in direction im2->im1
        # and we want to be consistent with optical flow from videos
        return im2, im1, flow, grid, kp2, kp1

    def warp_keypoints(self, keypoints, grid_unnormalized):
        from scipy.spatial.kdtree import KDTree
        warp_grid = grid_unnormalized.reshape(-1, 2)
        regular_grid = self.grid_pixels_unnormalized.reshape(-1, 2)
        kd = KDTree(warp_grid)
        dists, idxs = kd.query(keypoints)
        new_keypoints = regular_grid[idxs]
        return new_keypoints


class WarperSingle(object):
    returns_pairs = False

    def __init__(self, H, W, warpsd_all=0.0005, warpsd_subset=0.0, transsd=0.02,
                 scalesd=0.02, rotsd=2):
        self.H = H
        self.W = W
        self.warpsd_all = warpsd_all
        self.warpsd_subset = warpsd_subset
        self.transsd = transsd
        self.scalesd = scalesd
        self.rotsd = rotsd

        self.npixels = H * W
        self.nc = 10
        self.nctrlpts = self.nc * self.nc

        self.grid_pixels = tps_grid(H, W)
        self.grid_pixels_unnormalized = grid_unnormalize(self.grid_pixels.reshape(1, H, W, 2), self.H, self.W)
        self.grid_ctrlpts = tps_grid(self.nc, self.nc)
        self.U_ctrlpts = tps_U(self.grid_ctrlpts, self.grid_ctrlpts)
        self.U_pixels_ctrlpts = tps_U(self.grid_pixels, self.grid_ctrlpts)
        self.F = torch.cat((self.U_pixels_ctrlpts, torch.ones(self.npixels, 1), self.grid_pixels), 1)

    def __call__(self, im1, keypts=None, crop=0):
        kp1 = 0

        unsqueezed = False
        if len(im1.shape) == 3:
            im1 = im1.unsqueeze(0)
            unsqueezed = True

        assert im1.shape[0] == 1

        a = 1
        weights1 = random_tps_weights(self.nctrlpts, a * self.warpsd_all, a * self.warpsd_subset, a * self.transsd,
                                      a * self.scalesd, a * self.rotsd)

        grid1 = torch.matmul(self.F, weights1).reshape(1, self.H, self.W, 2)
        grid1_unnormalized = grid_unnormalize(grid1, self.H, self.W)
        if keypts is not None:
            kp1 = self.warp_keypoints(keypts, grid1_unnormalized)

        im1 = F.grid_sample(im1, grid1, align_corners = True)

        if crop != 0:
            im1 = im1[:, :, crop:-crop, crop:-crop]

        if unsqueezed:
            im1 = im1.squeeze(0)

        if crop != 0 and keypts is not None:
            kp1 -= crop

        # Reverse the order because due to inverse warping the "flow" is in direction im2->im1
        # and we want to be consistent with optical flow from videos
        return im1, kp1

    def warp_keypoints(self, keypoints, grid_unnormalized):
        from scipy.spatial.kdtree import KDTree
        warp_grid = grid_unnormalized.reshape(-1, 2)
        regular_grid = self.grid_pixels_unnormalized.reshape(-1, 2)
        kd = KDTree(warp_grid)
        dists, idxs = kd.query(keypoints)
        new_keypoints = regular_grid[idxs]
        return new_keypoints
