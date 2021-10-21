CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* (Charles) Zixin Zhang
* Tested on: Windows 11, 11th Core i7, 3060 Laptop GPU

# Result

![](comp.png)

- This denoiser is achieved by implementing the paper "Edge-Avoiding A-Trous Wavelet Transform for fast Global Illumination Filtering," by Dammertz, Sewtz, Hanika, and Lensch. 
- The above images is created using 
  - filter size = 80
  - Color weight = 38
  - Normal weight = 0.35
  - Position weight = 0.2

# Performance Analysis

 In my implementation, denoising is performed during the last iteration. For the simple cornell box, the denoising adds around 52ms to the render. 

- how much time denoising adds to your renders
- how denoising influences the number of iterations needed to get an "acceptably smooth" result
- how denoising at different resolutions impacts runtime
- how varying filter sizes affect performance



# References

- [_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
- [_Edge-Avoiding A-Trous Wavelet Transform for fast Global Illumination Filtering by Dammertz, Sewtz, Hanika, and Lensch_](https://jo.dreggn.org/home/2010_atrous.pdf)
- [_CUDA Denoiser by Janine Liu_](https://github.com/j9liu/Project4-CUDA-Denoiser)
- [_CUDA Pathtracing Denoiser by Jilin Liu_](https://github.com/Songsong97/Project4-CUDA-Denoiser)



