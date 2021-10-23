CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Xiao Wei
* Tested on: Windows 10, i9-9900Kh @ 3.6GHz 16.0GB, RTX 2080 Super, 16GB



Rendering Results
======================

50 Iteration Without Denoising | 50 Iteration with Denoising
------------ | -------------
![6ecf6a65fa2e822d428ae2b62947578](https://user-images.githubusercontent.com/66859615/138532684-c7a019d5-5482-472d-b865-732d96131e9a.png) | ![85ade895d32c3c9bf0a4239a4c08dbf](https://user-images.githubusercontent.com/66859615/138534724-69002dca-f533-4361-8d0e-95baf427cb86.png)


Analysis
======================

## Denoise Time
As we can see from the screen shot, the denoise takes some constant time in the rendering process. The average time needed is around 2.7ms

![fbbae417e8d1603946c373867a7ae61](https://user-images.githubusercontent.com/66859615/138532918-3c595b6c-b77c-42cd-b30d-9054b4c7711d.png)



## Acceptably smooth result
500 Iteration Without Denoising | 47 Iteration with Denoising
------------ | -------------
![b7ad8c9de6d039bad7efbbf22f447d5](https://user-images.githubusercontent.com/66859615/138535085-183777f6-f2c3-4f41-af69-f07dc8b46127.png) | ![fb80df8605d6cd029d433c9ab6b074b](https://user-images.githubusercontent.com/66859615/138535097-e56259e4-3e53-4cac-8d2b-fa8e7c657e1a.png)

29 Iteration with Denoising | 14 Iteration with Denoising
------------ | -------------
 ![52d5875411e3f04036a48c4151bab50](https://user-images.githubusercontent.com/66859615/138535119-32c72f6c-5d36-4012-b390-d842ec6091b8.png)| ![288388cc8d47bae7c75220bc90c7c62](https://user-images.githubusercontent.com/66859615/138535125-18143649-2b2d-4572-a030-d466591b9a5e.png)
 
 I did some experiment with different iterations. With proper arguments tuning for different iterations, we can find that we can get some almost perfect results with only 49 iterations and the result with 29 iterations is still ACCEPTABLE! However, with only 14 iteration, there are visible darker areas on the wall which is not that ideal. My answer is around 30 iterations.



## Average Denoise time with resolution
100 * 100 | 200 * 200 | 400 * 400 | 800 * 800
------------ | -------------|------------ | -------------
0.107ms | 0.19831ms | 0.639941ms | 2.94923ms

As we can see, the time taken to denoise the image increases as the resolution increase. It approximately proportional to the number of pixels (not so from 100 to 200 though)

## Average Denoise time with varying filter size 
10 | 20 | 40 | 80
------------ | -------------|------------ | -------------
1.59ms | 1.52ms | 2.268ms | 2.90s

The average to denoise the image increases as filter size increase and probably proportional to the LOG of filter size because of my implementation.


## Quality with different filter size and material
filter size 34 | filter size 107
------------ | -------------
![9f8286095d511b0bc9880b09b81ee73](https://user-images.githubusercontent.com/66859615/138536986-c7fb3621-0b8b-4309-aca7-6f0a33721d00.png)| ![0724274c2e7529096914fef86fb3b72](https://user-images.githubusercontent.com/66859615/138536989-812dfa0c-3553-4caf-b8c2-8e1482ca6a73.png)

Generally, larger filter size will blur the image more but at the same time reduce the visual glitch(darker area) more (quality increased on walls). In my case, there is sometimes too much blur when the filter is too large on the mirror sphere because the edge and position information are not conveyed to the mirror and the boundaries will be blurred. Thus, I think the method is more compatible with material with uniform color or less change. 

## Ceiling Light vs Cornell 
![af81d6999ca45d8935d5754704a1d0b](https://user-images.githubusercontent.com/66859615/138537444-68431a65-87a8-4b42-86d0-80ae7958bcba.png)
![7928687be204a83fc3d0543b74caf9e](https://user-images.githubusercontent.com/66859615/138537447-51e7f7e6-d410-48d4-ac73-b020ba3855c6.png)

With same number of iteration, the Ceiling light scene produce much better result than Cornell. For Cornell, the smaller light size make it harder to converge. The path segments have smaller chance to intersect with light in their lifespan and thus produce more noise.
