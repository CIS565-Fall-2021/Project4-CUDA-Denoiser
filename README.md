CUDA Denoiser For CUDA Path Tracer
==================================

<img src="img\video.gif" height="500px"/>

|         Before denoising (2 samples)          |                     After denoising                      |
| :-------------------------------------------: | :------------------------------------------------------: |
| <img src="img\cornell-2.png" height="300px"/> | <img src="img\cornell-denoised-2sp.png" height="300px"/> |

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Anthony Mansur
  - https://www.linkedin.com/in/anthony-mansur-ab3719125/
- Tested on: Windows 10, AMD Ryzen 5 3600, Geforce RTX 2060 Super (personal)

## Introduction

One of the main challenges to ray tracing is getting the full picture without the noise. When using naïve implementations, this usually takes many iterations of the algorithm, which quickly becomes an issue for real-time applications.  As using the GPU has already allowed us to get great results in near real-time, see the previous [project]("https://github.com/anthonymansur/Project3-CUDA-Path-Tracer"),  our goal now is limit the amount of iterations needed to get an image without the noise. 

The way to do this is with the use of denoisers. Denoisers help produce a smoother appearance in a pathtraced image with fewer samples-per-pixel/iterations. This is done by blurring pixels. However, a simple blur is usually never what we want. For instance, take a look at this simple cornell box below and see the effect of applying a gaussian blur.

| Cornell box (10 samples)                       |            Cornell box (5000 samples)            |               Cornell box (simple blur)                |
| ---------------------------------------------- | :----------------------------------------------: | :----------------------------------------------------: |
| <img src="img\cornell-10.png" height="300px"/> | <img src="img\cornell-5000.png" height="300px"/> | <img src="img\cornell-10-blurred.png" height="300px"/> |

Obviously it doesn't come quite close to the cornell box with 5000 samples, but we do see an improvement: less grainy-ness. However, this comes at a great cost as we lose the sharp edges as we blur. To solve this, we need a new filter that preserves the edges. Hence, we implement a technique mentioned in the paper ["Edge-Avoiding A-Trous Wavelet Transform for fast Global Illumination Filtering"](https://jo.dreggn.org/home/2010_atrous.pdf) that does just that.



## A-Trous Wavelet Transform Filtering

To understand how this algorithm works, we must understand how any filter, like the gaussian, works. For every pixel in our image, we look at the neighboring pixels, and sum the color value of each one, multiplied by a factor at every value, so that we end up we a normalized value. These factors can be stored in a matrix, commonly known as a kernel, allowing us to take the convolution and get the blurred image. A gaussian blur is simply the the kernel with values based on the gaussian distribution. The wider this kernel is, the more of a blurred effect we get.

|                  kernel size = 4x4                  |                  kernel size = 6x6                  |                  kernel size = 8x8                  |
| :-------------------------------------------------: | :-------------------------------------------------: | :-------------------------------------------------: |
| <img src="img\cornell-blur-16.png" height="300px"/> | <img src="img\cornell-blur-36.png" height="300px"/> | <img src="img\cornell-blur-64.png" height="300px"/> |

> note: in the paper and in this implementation, our kernel size is fixed to 5x5, however, we apply an offset iteratively using these same values to get the effect of a larger kernel. This helps us with performance (see the following section).

At the core of this algorithm, the denoiser uses what's called a *geometry buffer* to guide its smoothing filter. This buffer stores information about the scene, namely its color, position in space, and surface normal (See visualization below). As opposed to simply multiplying each pixel by the factor in the kernel, we also multiply by a *weight*. This weight is the derived by the multiplication of three other weights, one for the difference in the raytraced colors between the neighbor pixel and the center pixel, one for the difference in normal, and one for the difference in position. 

|                  per-pixel normals                  |           per-pixel positions (scaled down)           |
| :-------------------------------------------------: | :---------------------------------------------------: |
| <img src="img\cornell-normals.png" height="300px"/> | <img src="img\cornell-positions.png" height="300px"/> |



To get a better sense of what these weights do to our scene, below is a visualization of the effect of weighting for each individual weight.

|                Color-based blur                |                Normal-based blur                |                Position-based blur                |
| :--------------------------------------------: | :---------------------------------------------: | :-----------------------------------------------: |
| <img src="img\color-blur.png" height="300px"/> | <img src="img\normal-blur.png" height="300px"/> | <img src="img\position-blur.png" height="300px"/> |



With all three components combined, we get an image that looks like this:

| Cornell box (10 samples)                       |            Cornell box (5000 samples)            |              Cornell box (simple blur)               |
| ---------------------------------------------- | :----------------------------------------------: | :--------------------------------------------------: |
| <img src="img\cornell-10.png" height="300px"/> | <img src="img\cornell-5000.png" height="300px"/> | <img src="img\cornell-denoised.png" height="300px"/> |

Although there is definitely room for improvement, specifically on the parameter fine-tuning, we can already see the improvements that denoising gives us. We can even get great results with as low as 2 samples per path.

|         Before denoising (2 samples)          |                     After denoising                      |
| :-------------------------------------------: | :------------------------------------------------------: |
| <img src="img\cornell-2.png" height="300px"/> | <img src="img\cornell-denoised-2sp.png" height="300px"/> |

All of this, however, doesn't come for free. There is, of course, a performance and memory hit when using this algorithm. We have to now store an entire G-Buffer for the scene, and we have add this denoising step into our render loop. In most cases, however, by reducing the number of iterations needed to get an image that converges, ray tracing be greatly improved for real-time applications.



## Performance Analysis

To be done.
