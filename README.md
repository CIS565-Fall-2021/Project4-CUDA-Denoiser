CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Kaan Erdogmus
* Tested on: Windows 10, i7-8850H @ 2.2592GHz 24GB, Quadro P1000

### Denoising
The denoising checkbox was changed to a button for easier usage. The denoising is implemented as an A-trous kernel with weights based on the ray-traced color, normal, and position vectors. 

The A-trous kernel approximates a Gaussian blur and iterates by doubling the size of the area the kernel is applied to until the filter size is reached. The A-trous kernel is weighed through a weighted product (using inputted parameters) of the individual ray-traced, normal, and position weights. Each of these three weights are calculated as the exponent of the negation of the squared distance between the current point and a point corresponding to a kernel entry adjusted for the variance of the magnitudes of the rt, normal, or position vectors.

The variance is calculated in a single kernel on the GPU and a possible future optimization is to parallelize the calculation of the variance (by parallelizing the computation of the average and then parallelizing the computation of the sum of the deviations from the average squared).

### Denoising Analysis
The visual effects do not vary fully linearly with the filter sizes and the change becomes negligible beyond the halfway point.
The method was overall more effective with the material of the sphere than the material of the walls, with the walls retaining particles and noisy fragments with the A-trous filter.
The cornell scene with a full ceiling of light produced considerably better denoised results than the cornell scene without. The primary reason is that, with a smaller light source, there's significantly more noise with the same number of iterations (as many more rays never reach the light source) and as such, the denoising is less useful.

Smaller filter sizes resulted in a minor boost to performance, but not highly noticeable. This denoising implementation appears to be much more effective at more noisy images than those with greater RT iterations but does not improve noisy images to those comparable to higher-iterated ones. As such, for an image to be "acceptably smooth". it still requires a large number of iterations, implying that the denoising implementation does not have a large effect on the number of iterations.


