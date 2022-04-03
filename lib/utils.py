import os
import re
import sys

import cv2
import numpy as np
import torch

from torch.onnx import register_custom_op_symbolic
import torch.onnx.symbolic_helper as sym_help

# # Ref: https://github.com/pytorch/pytorch/issues/27212#issuecomment-1059773074
# def grid_sampler(g, input, grid, mode, padding_mode, align_corners):
#     # mode
#     #   'bilinear'      : onnx::Constant[value={0}]
#     #   'nearest'       : onnx::Constant[value={1}]
#     #   'bicubic'       : onnx::Constant[value={2}]
#     # padding_mode
#     #   'zeros'         : onnx::Constant[value={0}]
#     #   'border'        : onnx::Constant[value={1}]
#     #   'reflection'    : onnx::Constant[value={2}]
#     mode = sym_help._maybe_get_const(mode, "i")
#     padding_mode = sym_help._maybe_get_const(padding_mode, "i")
#     mode_str = ['bilinear', 'nearest', 'bicubic'][mode]
#     padding_mode_str = ['zeros', 'border', 'reflection'][padding_mode]
#     align_corners = int(sym_help._maybe_get_const(align_corners, "b"))

#     return g.op("com.microsoft::GridSample", input, grid,
#                 mode_s=mode_str,
#                 padding_mode_s=padding_mode_str,
#                 align_corners_i=align_corners)
    
# register_custom_op_symbolic('::grid_sampler', grid_sampler, 1)

# Ref: https://zenn.dev/pinto0309/scraps/7d4032067d0160
def bilinear_grid_sample(im, grid, align_corners=False):
    """Given an input and a flow-field grid, computes the output using input
    values and pixel locations from grid. Supported only bilinear interpolation
    method to sample the input pixels.

    Args:
        im (torch.Tensor): Input feature map, shape (N, C, H, W)
        grid (torch.Tensor): Point coordinates, shape (N, Hg, Wg, 2)
        align_corners {bool}: If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.

    Returns:
        torch.Tensor: A tensor with sampled points, shape (N, C, Hg, Wg)
    """
    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.view(n, -1)
    y = y.view(n, -1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Apply default for grid_sample function zero padding
    im_padded =  torch.nn.functional.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)
    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size
    x0 = torch.where(x0 < 0, torch.tensor(0, device=im.device), x0)
    x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1, device=im.device), x0)
    x1 = torch.where(x1 < 0, torch.tensor(0, device=im.device), x1)
    x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1, device=im.device), x1)
    y0 = torch.where(y0 < 0, torch.tensor(0, device=im.device), y0)
    y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1, device=im.device), y0)
    y1 = torch.where(y1 < 0, torch.tensor(0, device=im.device), y1)
    y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1, device=im.device), y1)

    im_padded = im_padded.view(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

    Ia = torch.gather(im_padded, 2, x0_y0)
    Ib = torch.gather(im_padded, 2, x0_y1)
    Ic = torch.gather(im_padded, 2, x1_y0)
    Id = torch.gather(im_padded, 2, x1_y1)

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)

def scale_coords(points, max_length):
    return 2 * points / (max_length - 1.0) - 1.0


def to_numpy(tensor):
    return tensor.squeeze().detach().cpu().numpy()


def interpolate(feat, uv):
    uv = uv.transpose(1, 2)
    uv = uv.unsqueeze(2)
    # samples = torch.nn.functional.grid_sample(feat, uv)
    samples = bilinear_grid_sample(feat, uv)
    
    return samples[:, :, :, 0]

def load_ckp(checkpoint_path, cuda, model, optimizer):
    checkpoint = torch.load(checkpoint_path, map_location=cuda)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr = checkpoint["learning_rate"]
    return model, optimizer, checkpoint["epoch"], lr

def save_ckp(state, checkpoint_path):
    torch.save(state, checkpoint_path)


def readPFM(file):
    file = open(file, "rb")
    header = file.readline().rstrip()

    if (sys.version[0]) == "3":
        header = header.decode("utf-8")
    if header == "PF":
        color = True
    elif header == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    if (sys.version[0]) == "3":
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    else:
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    if (sys.version[0]) == "3":
        scale = float(file.readline().rstrip().decode("utf-8"))
    else:
        scale = float(file.readline().rstrip())

    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def disp_loader(path, scale_factor16bit=256):
    disp = None
    if not os.path.exists(path):
        raise ValueError("Cannot open disp: " + path)

    if path.endswith("pfm"):
        disp = np.expand_dims(readPFM(path), 0)
    if path.endswith("png"):
        disp = np.expand_dims(cv2.imread(path, -1), 0)
        if disp.dtype == np.uint16:
            disp = disp / float(scale_factor16bit)
    if path.endswith("npy"):
        disp = np.expand_dims(np.load(path, mmap_mode="c"), 0)
    if disp is None:
        raise ValueError("Problems while loading the disp")
    # Remove invalid values
    disp[np.isinf(disp)] = 0

    return disp.transpose(1, 2, 0).astype(np.float32)

def resize_imgs(imgs):
    dim = len(imgs[0].shape)
    if dim==3:
        assert(imgs[0].shape[0]==3)
        height = imgs[0].shape[1]
        width = imgs[0].shape[2]
    elif dim==2:
        height = imgs[0].shape[0]
        width = imgs[0].shape[1]
    else:
        raise RuntimeError("Unsupported dimension!")

    for idx in range(1, len(imgs)):
        if dim==3:
            imgs[idx] = cv2.resize(imgs[idx].transpose(1,2,0), (width, height)).transpose(2,0,1)
        else:
            imgs[idx] = cv2.resize(imgs[idx], (width, height), interpolation = cv2.INTER_NEAREST)
    return
    
def img_loader(path, mode="passive", height=2160, width=3840):
    img = None
    if not os.path.exists(path):
        raise ValueError(f"Cannot open image: {path}")
    if path.endswith("raw"):
        img = (
            np.fromfile(open(path, "rb"), dtype=np.uint8).reshape(height, width, 3)
            if mode == "passive"
            else np.fromfile(open(path, "rb"), dtype=np.uint8).reshape(height, width, 1)
        )
    else:
        img = cv2.imread(path,1)
        if img.ndim == 2:
            img = np.expand_dims(img, -1) #

    return img


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    return lr


def shift_2d_replace(data, dx, dy, constant=False):
    shifted_data = np.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data


def rgb_loader(path, height=2160, width=3840, channels=3):
    img = None
    try:
        if path.endswith("raw"):
            img = np.fromfile(open(path, "rb"), dtype=np.uint8).reshape(
                height, width, channels
            )
        else:
            img = cv2.imread(path, -1)
    except:
        print("Cannot open RGB image: " + path)

    return img


def pad_img(img: np.ndarray, height: int = 1024, width: int = 1024, divisor: int = 32):
    """Pad the input image, making it larger at least (:attr:`height`, :attr:`width`)

    Params:
    ----------

    img (np.ndarray):
        array with shape h x w x c

    height (int):
        new minimum height

    width (int):
        new minimum width

    divisor (int):
        divisor factor, it forces the padded array to be multiple of divisor

    Returns:
        a new array with shape  H x W x c, multiple of divisior, and
        the amount of padding
    """
    h_pad = 0 if (height % divisor) == 0 else divisor - (height % divisor)
    top = h_pad // 2
    bottom = h_pad - top
    w_pad = 0 if (width % divisor) == 0 else divisor - (width % divisor)
    left = w_pad // 2
    right = w_pad - left
    img = np.lib.pad(img, ((top, bottom), (left, right), (0, 0)), mode="reflect")
    pad = np.stack([top, bottom, left, right], axis=0)
    return img, pad


def depad_img(
    img: np.ndarray,
    pad: np.ndarray,
    upsampling_factor: float = 1,
):
    """Remove padding from tensor

    Params:
    -------------

    img (np.ndarray):
        array to de-pad, with shape CxHxW or HxW
    pad (np.ndarray):
        array (top_pad, bottom_pad, left_pad, right_pad) with shape 1x4
    upsampling_factor (int):
        how to scale crops. For instance, if :attr:`upsampling_factor: is 4,
        crops are upscaled by 4.
        Default is 1.
    Returns:
    ------------
        a np.ndarray
    """

    if not img.ndim == 3:
        img = np.expand_dims(img, 0)
    pad = pad.squeeze()
    top = int(pad[0] * upsampling_factor)
    bottom = int(pad[1] * upsampling_factor)
    left = int(pad[2] * upsampling_factor)
    right = int(pad[3] * upsampling_factor)

    return img[
        :,
        top : img.shape[1] - bottom,
        left : img.shape[2] - right,
    ]

def get_boundaries(disp, th=1.0, dilation=10):
    edges_y = np.logical_or(
        np.pad(np.abs(disp[1:, :] - disp[:-1, :]) > th, ((1, 0), (0, 0))),
        np.pad(np.abs(disp[:-1, :] - disp[1:, :]) > th, ((0, 1), (0, 0))),
    )
    edges_x = np.logical_or(
        np.pad(np.abs(disp[:, 1:] - disp[:, :-1]) > th, ((0, 0), (1, 0))),
        np.pad(np.abs(disp[:, :-1] - disp[:, 1:]) > th, ((0, 0), (0, 1))),
    )
    edges = np.logical_or(edges_y, edges_x).astype(np.float32)

    if dilation > 0:
        kernel = np.ones((dilation, dilation), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    return edges