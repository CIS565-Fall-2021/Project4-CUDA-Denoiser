from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
from skimage.color import rgb2gray
import argparse

# Compare difference between two images based on SSIM

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first_img", required=True, help="Directory of the first image")
ap.add_argument("-s", "--second_img", required=True, help="Directory of the second image")
args = vars(ap.parse_args())

image1 = imread(args["first_img"])
image2 = imread(args["second_img"])

gray1 = rgb2gray(image1)
gray2 = rgb2gray(image2)

print(ssim(gray1, gray2, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))