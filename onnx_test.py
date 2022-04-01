import cv2
import numpy as np
import onnxruntime

from performance_monitor import *

class Refiner():

	def __init__(self, model_path, max_dist=10):

		self.initialize_model(model_path, max_dist)

	def __call__(self, rgb_img, disp_img):

		return self.update(rgb_img, disp_img)

	def initialize_model(self, model_path, max_dist=10):

		self.max_dist = max_dist

		# Initialize model session
		self.session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider',
																		   'CPUExecutionProvider'])

		# Get model info
		self.get_input_details()
		self.get_output_details()

	def update(self, rgb_img, disp_img):

		rgb_tensor = self.prepare_input(rgb_img)
		disp_tensor = self.prepare_input(disp_img)

		# Estimate the disparity map
		outputs = self.inference(rgb_tensor, disp_tensor)
		self.disparity_map, _ = self.process_output(outputs)

		# # Estimate depth map from the disparity
		# self.depth_map = self.get_depth_from_disparity(self.disparity_map, self.camera_config)

		return self.disparity_map

	def prepare_input(self, img):

		# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		self.img_height, self.img_width = img.shape[:2]

		img_input = cv2.resize(img, (self.input_width,self.input_height))

		dim = len(img_input.shape)
		if dim < 3:
			img_input = np.expand_dims(img_input, -1)

		# # Scale input pixel values to -1 to 1
		# mean=[0.485, 0.456, 0.406]
		# std=[0.229, 0.224, 0.225]
		
		# img_input = ((img_input/ 255.0 - mean) / std)
		print(img_input.shape, self.input_width,self.input_height)
		img_input = img_input.transpose(2, 0, 1)
		img_input = img_input[np.newaxis,:,:,:]        

		return img_input.astype(np.float32)

	@performance
	def inference(self, rgb_tensor, disp_tensor):

		outputs = self.session.run(self.output_names, {self.input_names[0]: rgb_tensor,
													   self.input_names[1]: disp_tensor})

		return outputs

	def process_output(self, outputs): 
		return np.squeeze(outputs[0]), np.squeeze(outputs[1])

	def get_input_details(self):

		model_inputs = self.session.get_inputs()
		self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

		self.input_shape = model_inputs[0].shape
		self.input_height = self.input_shape[2]
		self.input_width = self.input_shape[3]

	def get_output_details(self):

		model_outputs = self.session.get_outputs()
		self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

		self.output_shape = model_outputs[0].shape

if __name__ == '__main__':

	import matplotlib.pyplot as plt
	from lib.utils import pad_img, depad_img, img_loader, disp_loader
 
	# Initialize model
	model_path = 'disp_refiner_sim.onnx'
	disp_refiner = Refiner(model_path)

	rgb_img = img_loader("sample/rgb_middlebury.png")
	disp_img = disp_loader("sample/disp_middlebury.png", 256)
	
	# Estimate depth and colorize it
	disparity_map = disp_refiner(rgb_img, disp_img)

	plt.imshow(disparity_map, cmap="magma")
	plt.show()
	plt.imsave("output_onnx.png", disparity_map, cmap="magma")
