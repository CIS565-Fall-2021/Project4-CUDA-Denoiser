CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Yuxuan Zhu
  * [LinkedIn](https://www.linkedin.com/in/andrewyxzhu/)
* Tested on: Windows 10, i7-7700HQ @ 2.80GHz 16GB, GTX 1050 4096MB (Personal Laptop)

## Demo

![Demo](img/final.jpg)

## Performance Analysis

**Qualitative Comparison**

Original (1/2/4/8/16 iter) |  Denoised (1/2/4/8/16 iter)
:-------------------------:|:-------------------------:
![1iter](img/1iter.jpg)  |  ![1iterDenoise](img/1iterDenoise.jpg



## Bloopers

![Blooper](img/blooper1.jpg)

This image looks like it has been terribly exposed and it is caused by the program not normalziing the ray traced image by the number of iterations.




Loop unrolling

Do deniosing once after certain iterations more efficient doing it multiple times is useless.
using constant memory


iteratively updated offset

position need to divide by 10 cuz not normalized.
smaller more blurry