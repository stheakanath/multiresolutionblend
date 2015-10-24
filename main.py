import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import scipy
import cv2
from align_image_code import align_images
from PIL import Image
import sys

# Matlab's rgb2gray by Stackoverflow: http://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
def rgb2gray(rgb):
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	return gray

# Part 0
def unmask_filter(imname, alpha):
	im = cv2.imread(imname)
	blur = cv2.GaussianBlur(im, (5, 5), 0)
	unmask = ((im - blur) * alpha) + im
	title = "Unmask Filter | Alpha: " + str(alpha)
	plt.imshow(unmask),plt.title(title)
	plt.show()

# Helper function for Part 1
def hybrid_image(im1, im2, sigma1, sigma2):
	# Low pass filter
	low_pass = scipy.ndimage.filters.gaussian_filter(im1, sigma1)
	plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(low_pass)))))
	plt.show()
	high_pass = im2 - scipy.ndimage.filters.gaussian_filter(im2, sigma2)
	plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(high_pass)))))
	plt.show()
	return low_pass + high_pass

# Part 1 - Hybrid Images
def hybrid(im1_name, im2_name, sig1, sig2):
	# high sf
	im1 = plt.imread(im1_name)/255.

	# low sf
	im2 = plt.imread(im2_name)/255.

	# Next align images (this code is provided, but may be improved)
	im1_aligned, im2_aligned = align_images(im1, im2)

	im1_aligned = rgb2gray(im1_aligned)
	im2_aligned = rgb2gray(im2_aligned)

	orig = scipy.ndimage.filters.gaussian_filter(im1, 1)
	low_pass = scipy.ndimage.filters.gaussian_filter(im1, 5)
	high_pass = orig - low_pass
	final = low_pass + high_pass

	sigma1 = sig1
	sigma2 = sig2
	hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

	plt.imshow(hybrid, cmap="Greys_r")
	plt.show()

# Helper function for Part 2
def pyramids(im, n):
	# Gaussian Pyramids
	sigma = 1
	fig = plt.figure()
	j = 1
	for x in range(n + 1):
		a = fig.add_subplot(3, 2, j)
		img = scipy.ndimage.filters.gaussian_filter(im, sigma)
		plt.imshow(img, cmap="Greys_r")
		plt.axis('off')
		plt.title("Sigma: " + str(sigma))
		sigma *= 2
		j += 1
	plt.show()

	# Laplacian Pyramid
	sigma = 1
	fig = plt.figure()
	j = 1
	for x in range(n + 1):
		a = fig.add_subplot(3, 2, j)
		img = im - scipy.ndimage.filters.gaussian_filter(im, sigma)
		plt.imshow(img, cmap="Greys_r")
		plt.axis('off')
		plt.title("Sigma: " + str(sigma))
		sigma *= 2
		j += 1
	plt.show()


# Part 2  - Gaussian and Laplacian Stacks
def stacks(im_name, n):
	im = rgb2gray(plt.imread(im_name) / 255.)
	pyramids(im, n)

# Helper function for Part 3. Constructs and Deconstructs pyramids
def pyramids(image, mask=False, reconstruct=False):
	kernal = np.array(((1.0/256, 4.0/256,  6.0/256,  4.0/256,  1.0/256), (4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256), (6.0/256, 24.0/256, 36.0/256, 24.0/256, 6.0/256), (4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256), (1.0/256, 4.0/256,  6.0/256,  4.0/256,  1.0/256)))
	if not reconstruct:
		G, L = [image], []
		while image.shape[0] >= 2 and image.shape[1] >= 2:
			image = scipy.signal.convolve2d(image, kernal, mode='same', fillvalue=1)[::2, ::2]
			G.append(image)
		for i in range(len(G) - 1):
			next_img = np.zeros((2 * G[i + 1].shape[0], 2 * G[i + 1].shape[1]))
			next_img[::2, ::2] = G[i + 1]
			L.append(G[i] - scipy.signal.convolve2d(next_img, 4 * kernal, mode='same', fillvalue=1))
		return G[:-1] if mask else L
	else:
		for i in range(len(image)):
			for j in range(i):
				next_img = np.zeros((2 * image[i].shape[0], 2 * image[i].shape[1]))
				next_img[::2, ::2] = image[i]
				image[i] = scipy.signal.convolve2d(next_img, 4 * kernal, mode='same', fillvalue=1)
		tot_sum = np.sum(image, axis=0)
		tot_sum[tot_sum < 0.0] = 0.0
		tot_sum[tot_sum > 255.0] = 255.0
		return tot_sum

# Helper function, follows formula described in paper
def blend_pyramids(im1_pyramid, im2_pyramid, mask_pyramid):
	blended = []
	for i in range(len(mask_pyramid)):
		blended.append(im1_pyramid[i] * (1.0 - mask_pyramid[i]) + im2_pyramid[i] * mask_pyramid[i])
	return blended

# Part 3 - Multiresolution Blending, with Color
def blend(im1_name, im2_name, mask_name):
	im1, im2, mask = np.array(Image.open(im1_name)), np.array(Image.open(im2_name)), np.array(Image.open(mask_name)) / 255.0
	im1_pyramids, im2_pyramids, mask_pyramids = [], [], []
	for x in range(3):
		im1_pyramids.append(pyramids(im1[:,:,x]))
		im2_pyramids.append(pyramids(im2[:,:,x]))
		mask_pyramids.append(pyramids(mask[:,:,x], mask=True))
	
	final = np.empty((im1.shape[0], im1.shape[1], 3))
	color_blends = []
	for x in range(3):
		final[:,:,x] = pyramids(blend_pyramids(im1_pyramids[x], im2_pyramids[x], mask_pyramids[x]), reconstruct=True) / 255.0
	plt.imshow(final)
	plt.show()

# unmask_filter('campanile.jpg', 5)
# hybrid('sony.jpg', 'ankit.jpg', 10, 5)
# stacks('lincoln.jpg', 5)
blend(sys.argv[1], sys.argv[2], sys.argv[3])
