import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lib.model import Refiner
from lib.options import BaseOptions
from lib.utils import pad_img, depad_img, img_loader, disp_loader
 
# get options
opt = BaseOptions()
opt.backbone = "vgg13"
opt.max_disp = 256
opt.num_in_ch = 4 
opt.upsampling_factor = 1
opt.downsampling_factor = 1 
opt.disp_scale = 1
opt.scale_factor16bit = 256

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu" # TODO: Comment for cuda, but it give memory issues when using bilinear_grid_sample() function
opt.device = device

# When converting the model to ONNX, if True, it removes the loop in the forward function
# which removes the scatterND nodes of the model. However, this takes more GPU memory which limits 
# the input resolution
opt.convert_without_scatternd = False 

input_shape = (640, 480) # Make multiple of 32 to avoid having to use pads. It allows the points calculation inside the model

model_path = "checkpoints/sceneflow/net_latest"
img_path = "sample/rgb_middlebury.png"
disp_path = "sample/disp_middlebury.png"

# Read data
rgb = img_loader(img_path)
rgb = cv2.resize(rgb, input_shape)
height, width = rgb.shape[:2]

disp = disp_loader(disp_path, opt.scale_factor16bit) / opt.disp_scale
disp[disp > opt.max_disp] = 0
height_disp, width_disp = disp.shape[:2]
disp = np.expand_dims(cv2.resize(disp, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST),-1)
rgb, pad = pad_img(rgb, height=height, width=width, divisor=32)
disp, _ = pad_img(disp, height=height, width=width, divisor=32)

rgb = torch.from_numpy(rgb).float()
disp = torch.from_numpy(disp).float()
o_shape = torch.from_numpy(np.asarray((height, width)))

rgb = rgb.permute(2, 0, 1)
disp = disp.permute(2, 0, 1)

# create net
net = Refiner(opt).to(device=device)
net.load_state_dict(torch.load(model_path, map_location=device)["state_dict"])
net.eval()

disp_tensor = disp.to(device=device)
img_tensor = rgb.to(device=device)

if img_tensor.ndim == 3:
    img_tensor = torch.unsqueeze(img_tensor, 0)
if disp_tensor.ndim == 3:
    disp_tensor = torch.unsqueeze(disp_tensor, 0)

height=img_tensor[0].shape[1]
width=img_tensor[0].shape[2]

start_x=0
start_y=0
end_x=width
end_y=height

nx = np.linspace(start_x, end_x, width)
ny = np.linspace(start_y, end_y, height)
u, v = np.meshgrid(nx, ny)
coords = np.expand_dims(np.stack((u.flatten(), v.flatten()), axis=-1), 0)
coords = torch.Tensor(coords).float().to(device=opt.device)
coords = coords.reshape(1, -1, 2)

torch.onnx.export(net,               # model being run
              (img_tensor, disp_tensor, coords),                         # model input (or a tuple for multiple inputs)
              "disp_refiner.onnx",   # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=11,          # the ONNX version to export the model to
              do_constant_folding=True,
              input_names = ['rgb', 'disp', 'points'],   # the model's input names
              output_names = ['refined_disp', 'conf'], # the model's output names
              )

np.save("points.npy", coords.detach().cpu().numpy())

pred, confidence = net(img_tensor, disp_tensor, coords)
pred = pred.squeeze().detach().cpu().numpy()
confidence = confidence.squeeze().detach().cpu().numpy()

plt.imsave("output_pytorch.png", pred, cmap="magma")

