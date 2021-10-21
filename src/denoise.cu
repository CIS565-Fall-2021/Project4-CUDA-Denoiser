#include "denoise.h"

namespace Denoise {

__constant__ float denoiseKernel[5][5];

glm::vec3* dev_tmpBuffer = nullptr;
#if !PREGATHER_FINAL_IMAGE
glm::vec3* dev_tmpImageTextureBuffer = nullptr;
#endif // PREGATHER_FINAL_IMAGE

glm::vec3* dev_lastFrameBuffer = nullptr;
GBufferPixel* dev_lastGBuffer = nullptr;

glm::vec3* dev_misc = nullptr;
ui8* dev_valid = nullptr;

void init(glm::ivec2 size) {
    float tmpDenoiseKernel1D[] = { 1.f / 16.f, 1.f / 4.f, 3.f / 8.f, 1.f / 4.f, 1.f / 16.f };
    float tmpDenoiseKernel2D[5][5];
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            tmpDenoiseKernel2D[i][j] = tmpDenoiseKernel1D[i] * tmpDenoiseKernel1D[j];
        }
    }
    cudaMemcpyToSymbol(denoiseKernel, tmpDenoiseKernel2D, sizeof(float) * 5 * 5, 0, cudaMemcpyHostToDevice);
    cudaMalloc(&dev_tmpBuffer, sizeof(glm::vec3) * size.x * size.y);
#if !PREGATHER_FINAL_IMAGE
    cudaMalloc(&dev_tmpImageTextureBuffer, sizeof(glm::vec3) * size.x * size.y);
#endif // PREGATHER_FINAL_IMAGE
    cudaMalloc(&dev_lastFrameBuffer, sizeof(glm::vec3) * size.x * size.y);
    cudaMalloc(&dev_lastGBuffer, sizeof(GBufferPixel) * size.x * size.y);
    cudaMalloc(&dev_misc, sizeof(glm::vec3) * size.x * size.y);
    cudaMalloc(&dev_valid, sizeof(ui8) * size.x * size.y);
}

void deinit() {
    cudaFree(dev_valid);
    cudaFree(dev_misc);
    cudaFree(dev_lastGBuffer);
    cudaFree(dev_lastFrameBuffer);
#if !PREGATHER_FINAL_IMAGE
    cudaFree(dev_tmpImageTextureBuffer);
#endif // PREGATHER_FINAL_IMAGE
    cudaFree(dev_tmpBuffer);
}

void denoise(
    DenoiserType type, const dim3 blocksPerGrid2d, const dim3 blockSize2d, Texture2D<glm::vec3> outFrameBuffer, Texture2D<glm::vec3> inFrameBuffer, Texture2D<GBufferPixel> gBuffer, DenoiseWeightParam wParam,
    const Camera& camera, const Camera& lastCamera, DenoiseTemporalParam tParam) {
    switch (type) {
    case DenoiserType::A_TROUS:
        denoise_A_Trous(blocksPerGrid2d, blockSize2d, outFrameBuffer, inFrameBuffer, wParam);
        break;
    case DenoiserType::A_TROUS_EDGE_AVOIDING:
        denoise_A_Trous_EdgeAvoiding(blocksPerGrid2d, blockSize2d, outFrameBuffer, inFrameBuffer, gBuffer, wParam);
        break;
    case DenoiserType::A_TROUS_EDGE_AVOIDING_MORE_PARAM:
        denoise_A_Trous_EdgeAvoidingMoreParam(blocksPerGrid2d, blockSize2d, outFrameBuffer, inFrameBuffer, gBuffer, wParam);
        break;
    default:
        cudaMemcpy(outFrameBuffer.buffer, inFrameBuffer.buffer, sizeof(glm::vec3) * outFrameBuffer.size.x * outFrameBuffer.size.y, cudaMemcpyDeviceToDevice);
        break;
    }
    checkCUDAError("denoise");

    if (tParam.enable) {
        if (!tParam.init) {
            denoise_temporalAccumulation(blocksPerGrid2d, blockSize2d, outFrameBuffer, gBuffer, camera, lastCamera, tParam);
        }

        cudaMemcpy(dev_lastFrameBuffer, outFrameBuffer.buffer, sizeof(glm::vec3) * outFrameBuffer.size.x * outFrameBuffer.size.y, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_lastGBuffer, gBuffer.buffer, sizeof(GBufferPixel) * gBuffer.size.x * gBuffer.size.y, cudaMemcpyDeviceToDevice);
        checkCUDAError("temporal accumulation");
    }
}

__global__ void denoise_A_Torus_OneSweep(Texture2D<glm::vec3> outFrameBuffer, const Texture2D<glm::vec3> inFrameBuffer, int stride) {
    int idxX = blockDim.x * blockIdx.x + threadIdx.x;
    int idxY = blockDim.y * blockIdx.y + threadIdx.y;

    int sizeX = outFrameBuffer.size.x, sizeY = outFrameBuffer.size.y;

#if DENOISE_WITH_SHARED_MEMORY
    extern __shared__ glm::vec3 sharedFrameBuffer[]; // (blockDim.x + 4 * stride) * (blockDim.y + 4 * stride)
#endif // DENOISE_WITH_SHARED_MEMORY

    float totalWeight = 0.f;
    glm::vec3 color;
    if (idxX < sizeX && idxY < sizeY) {
#if DENOISE_WITH_SHARED_MEMORY
        //printf("before sync\n");
        for (int offsetY = threadIdx.y; offsetY < blockDim.y + 4 * stride; offsetY += blockDim.y) {
            int globalY = blockDim.y * blockIdx.y + offsetY - 2 * stride;
            if (globalY < 0 || globalY >= sizeY) {
                continue;
            }
            for (int offsetX = threadIdx.x; offsetX < blockDim.x + 4 * stride; offsetX += blockDim.x) {
                int globalX = blockDim.x * blockIdx.x + offsetX - 2 * stride;
                if (globalX < 0 || globalX >= sizeX) {
                    continue;
                }
                int globalIdx = Texture2D<glm::vec3>::index2Dto1D(outFrameBuffer.size, globalY, globalX);
                int smIdx = (offsetY) * (blockDim.x + 4 * stride) + (offsetX);
                sharedFrameBuffer[smIdx] = inFrameBuffer.buffer[globalIdx];
            }
        }
        __syncthreads();
        //printf("sync complete\n");
#endif // DENOISE_WITH_SHARED_MEMORY
        for (int dy = -2; dy <= 2; ++dy) {
            int y = idxY + dy * stride;
            if (y < 0 || y >= sizeY) {
                continue;
            }
            for (int dx = -2; dx <= 2; ++dx) {
                int x = idxX + dx * stride;
                if (x < 0 || x >= sizeX) {
                    continue;
                }

                float weight = denoiseKernel[dy + 2][dx + 2];
                totalWeight += weight;
#if DENOISE_WITH_SHARED_MEMORY
                int smY = threadIdx.y + (dy + 2) * stride;
                int smX = threadIdx.x + (dx + 2) * stride;
                int smIdx = smY * (blockDim.x + 4 * stride) + smX;
                color += sharedFrameBuffer[smIdx] * weight;
#else // DENOISE_WITH_SHARED_MEMORY
                color += inFrameBuffer.getPixelByHW(y, x) * weight;
#endif // DENOISE_WITH_SHARED_MEMORY
            }
        }
    }
    outFrameBuffer.setPixelByHW(idxY, idxX, totalWeight == 0.f ? glm::vec3(0.f) : color / totalWeight);
}

void denoise_A_Trous(const dim3 blocksPerGrid2d, const dim3 blockSize2d, Texture2D<glm::vec3> outFrameBuffer, Texture2D<glm::vec3> inFrameBuffer, DenoiseWeightParam wParam) {
    cudaMemcpy(dev_tmpBuffer, inFrameBuffer.buffer, sizeof(glm::vec3) * outFrameBuffer.size.x * outFrameBuffer.size.y, cudaMemcpyDeviceToDevice);
    Texture2D<glm::vec3> tmpBuffer;
    tmpBuffer.buffer = dev_tmpBuffer;
    tmpBuffer.size = inFrameBuffer.size;

    if (wParam.filterSize == 0) {
        cudaMemcpy(outFrameBuffer.buffer, inFrameBuffer.buffer, sizeof(glm::vec3) * outFrameBuffer.size.x * outFrameBuffer.size.y, cudaMemcpyDeviceToDevice);
    }

    for (int stride = 1; stride <= wParam.filterSize; stride <<= 1) {
        if (stride > 1) {
            cudaMemcpy(dev_tmpBuffer, outFrameBuffer.buffer, sizeof(glm::vec3) * outFrameBuffer.size.x * outFrameBuffer.size.y, cudaMemcpyDeviceToDevice);
        }
#if DENOISE_WITH_SHARED_MEMORY
        denoise_A_Torus_OneSweep<<<blocksPerGrid2d, blockSize2d, sizeof(glm::vec3) * (blockSize2d.x + 4 * stride) * (blockSize2d.y + 4 * stride)>>>(outFrameBuffer, tmpBuffer, stride);
#else // DENOISE_WITH_SHARED_MEMORY
        denoise_A_Torus_OneSweep<<<blocksPerGrid2d, blockSize2d>>>(outFrameBuffer, tmpBuffer, stride);
#endif // DENOISE_WITH_SHARED_MEMORY
    }
}

inline __device__ float getLogColorWeight(const glm::vec3& color0, const glm::vec3& color1, float colorSigmaSquare) {
    return colorSigmaSquare == 0.f ? 0.f : -glm::length(color1 - color0) / colorSigmaSquare;
}

inline __device__ float getLogNormalWeight(const glm::vec3& normal0, const glm::vec3& normal1, float normalSigmaSquare) {
    return normalSigmaSquare == 0.f ? 0.f : -glm::length(normal1 - normal0) / normalSigmaSquare;
}

inline __device__ float getLogPositionWeight(const glm::vec3& position0, const glm::vec3& position1, float positionSigmaSquare) {
    return positionSigmaSquare == 0.f ? 0.f : -glm::length(position1 - position0) / positionSigmaSquare;
}

__global__ void denoise_A_Torus_EdgeAvoiding_OneSweep(Texture2D<glm::vec3> outFrameBuffer, const Texture2D<glm::vec3> inFrameBuffer, const Texture2D<GBufferPixel> gBuffer, DenoiseWeightParam wParam, int stride) {
    int idxX = blockDim.x * blockIdx.x + threadIdx.x;
    int idxY = blockDim.y * blockIdx.y + threadIdx.y;

    int sizeX = outFrameBuffer.size.x, sizeY = outFrameBuffer.size.y;

#if DENOISE_WITH_SHARED_MEMORY
    extern __shared__ ui8 shMem[];
    glm::vec3* sharedFrameBuffer = reinterpret_cast<glm::vec3*>(shMem);
    GBufferPixel* sharedGBuffer = reinterpret_cast<GBufferPixel*>(shMem + sizeof(glm::vec3) * (blockDim.x + 4 * stride) * (blockDim.y + 4 * stride));
#endif // DENOISE_WITH_SHARED_MEMORY

    float totalWeight = 0.f;

    glm::vec3 color;
    if (idxX < sizeX && idxY < sizeY) {
#if DENOISE_WITH_SHARED_MEMORY
        //printf("before sync\n");
        for (int offsetY = threadIdx.y; offsetY < blockDim.y + 4 * stride; offsetY += blockDim.y) {
            int globalY = blockDim.y * blockIdx.y + offsetY - 2 * stride;
            if (globalY < 0 || globalY >= sizeY) {
                continue;
            }
            for (int offsetX = threadIdx.x; offsetX < blockDim.x + 4 * stride; offsetX += blockDim.x) {
                int globalX = blockDim.x * blockIdx.x + offsetX - 2 * stride;
                if (globalX < 0 || globalX >= sizeX) {
                    continue;
                }
                int globalIdx = Texture2D<glm::vec3>::index2Dto1D(outFrameBuffer.size, globalY, globalX);
                int smIdx = (offsetY) * (blockDim.x + 4 * stride) + (offsetX);
                sharedFrameBuffer[smIdx] = inFrameBuffer.buffer[globalIdx];
                sharedGBuffer[smIdx] = gBuffer.buffer[globalIdx];
            }
        }
        __syncthreads();
        GBufferPixel gbp0 = sharedGBuffer[threadIdx.y * (blockDim.x + 4 * stride) + threadIdx.x];
#else // DENOISE_WITH_SHARED_MEMORY
        GBufferPixel gbp0 = gBuffer.getPixelByHW(idxY, idxX);
#endif // DENOISE_WITH_SHARED_MEMORY
        for (int dy = -2; dy <= 2; ++dy) {
            int y = idxY + dy * stride;
            if (y < 0 || y >= sizeY) {
                continue;
            }
            for (int dx = -2; dx <= 2; ++dx) {
                int x = idxX + dx * stride;
                if (x < 0 || x >= sizeX) {
                    continue;
                }

#if DENOISE_WITH_SHARED_MEMORY
                int smY = threadIdx.y + (dy + 2) * stride;
                int smX = threadIdx.x + (dx + 2) * stride;
                int smIdx = smY * (blockDim.x + 4 * stride) + smX;
                GBufferPixel gbp1 = sharedGBuffer[smIdx];
#else // DENOISE_WITH_SHARED_MEMORY
                GBufferPixel gbp1 = gBuffer.getPixelByHW(y, x);
#endif // DENOISE_WITH_SHARED_MEMORY

                float weight = denoiseKernel[dy + 2][dx + 2];

                float logColorWeight = getLogColorWeight(gbp0.baseColor, gbp1.baseColor, wParam.colorWeight * wParam.colorWeight);
                float logNormalWeight = getLogNormalWeight(gbp0.surfaceNormal, gbp1.surfaceNormal, wParam.normalWeight * wParam.normalWeight);
                float logPositionWeight = getLogPositionWeight(gbp0.position, gbp1.position, wParam.positionWeight * wParam.positionWeight);

                if (y != idxY && x != idxX && (wParam.colorWeight == 0.f || wParam.normalWeight == 0.f || wParam.positionWeight == 0.f || gbp0.geometryId < 0)) {
                    continue;
                }

                weight *= __expf(logColorWeight + logNormalWeight + logPositionWeight);

                totalWeight += weight;
#if DENOISE_WITH_SHARED_MEMORY
                color += sharedFrameBuffer[smIdx] * weight;
#else // DENOISE_WITH_SHARED_MEMORY
                color += inFrameBuffer.getPixelByHW(y, x) * weight;
#endif // DENOISE_WITH_SHARED_MEMORY
            }
        }
    }
    outFrameBuffer.setPixelByHW(idxY, idxX, totalWeight == 0.f ? glm::vec3(0.f) : color / totalWeight);
}

void denoise_A_Trous_EdgeAvoiding(const dim3 blocksPerGrid2d, const dim3 blockSize2d, Texture2D<glm::vec3> outFrameBuffer, Texture2D<glm::vec3> inFrameBuffer, Texture2D<GBufferPixel> gBuffer, DenoiseWeightParam wParam) {
    cudaMemcpy(dev_tmpBuffer, inFrameBuffer.buffer, sizeof(glm::vec3) * outFrameBuffer.size.x * outFrameBuffer.size.y, cudaMemcpyDeviceToDevice);
    Texture2D<glm::vec3> tmpBuffer;
    tmpBuffer.buffer = dev_tmpBuffer;
    tmpBuffer.size = inFrameBuffer.size;

    if (wParam.filterSize == 0) {
        cudaMemcpy(outFrameBuffer.buffer, inFrameBuffer.buffer, sizeof(glm::vec3) * outFrameBuffer.size.x * outFrameBuffer.size.y, cudaMemcpyDeviceToDevice);
    }

    for (int stride = 1; stride <= wParam.filterSize; stride <<= 1) {
        if (stride > 1) {
            cudaMemcpy(dev_tmpBuffer, outFrameBuffer.buffer, sizeof(glm::vec3) * outFrameBuffer.size.x * outFrameBuffer.size.y, cudaMemcpyDeviceToDevice);
        }
#if DENOISE_WITH_SHARED_MEMORY
        denoise_A_Torus_EdgeAvoiding_OneSweep<<<blocksPerGrid2d, blockSize2d, (sizeof(glm::vec3) + sizeof(GBufferPixel)) * (blockSize2d.x + 4 * stride) * (blockSize2d.y + 4 * stride)>>>(outFrameBuffer, tmpBuffer, gBuffer, wParam, stride);
#else // DENOISE_WITH_SHARED_MEMORY
        denoise_A_Torus_EdgeAvoiding_OneSweep<<<blocksPerGrid2d, blockSize2d>>>(outFrameBuffer, tmpBuffer, gBuffer, wParam, stride);
#endif // DENOISE_WITH_SHARED_MEMORY
        wParam.doHalf();
    }
}

inline __device__ float getLogPlaneWeight(const glm::vec3& position0, const glm::vec3& position1, const glm::vec3& normal0, float planeSigmaSquare) {
    if (planeSigmaSquare == 0.f) {
        return 0.f;
    }
    glm::vec3 diffPos = position1 - position0;
    float distPos = glm::length(diffPos);
    float distPlane = distPos == 0.f ? 0.f : glm::dot(normal0, diffPos / distPos);
    return -glm::abs(distPlane) / planeSigmaSquare;
}

__global__ void denoise_A_Torus_EdgeAvoidingMoreParam_OneSweep(Texture2D<glm::vec3> outFrameBuffer, const Texture2D<glm::vec3> inFrameBuffer, const Texture2D<GBufferPixel> gBuffer, DenoiseWeightParam wParam, int stride) {
    int idxX = blockDim.x * blockIdx.x + threadIdx.x;
    int idxY = blockDim.y * blockIdx.y + threadIdx.y;

    int sizeX = outFrameBuffer.size.x, sizeY = outFrameBuffer.size.y;

    float totalWeight = 0.f;

#if DENOISE_WITH_SHARED_MEMORY
    extern __shared__ ui8 shMem[];
    glm::vec3* sharedFrameBuffer = reinterpret_cast<glm::vec3*>(shMem);
    GBufferPixel* sharedGBuffer = reinterpret_cast<GBufferPixel*>(shMem + sizeof(glm::vec3) * (blockDim.x + 4 * stride) * (blockDim.y + 4 * stride));
#endif // DENOISE_WITH_SHARED_MEMORY

    glm::vec3 color;
    if (idxX < sizeX && idxY < sizeY) {
#if DENOISE_WITH_SHARED_MEMORY
        //printf("before sync\n");
        for (int offsetY = threadIdx.y; offsetY < blockDim.y + 4 * stride; offsetY += blockDim.y) {
            int globalY = blockDim.y * blockIdx.y + offsetY - 2 * stride;
            if (globalY < 0 || globalY >= sizeY) {
                continue;
            }
            for (int offsetX = threadIdx.x; offsetX < blockDim.x + 4 * stride; offsetX += blockDim.x) {
                int globalX = blockDim.x * blockIdx.x + offsetX - 2 * stride;
                if (globalX < 0 || globalX >= sizeX) {
                    continue;
                }
                int globalIdx = Texture2D<glm::vec3>::index2Dto1D(outFrameBuffer.size, globalY, globalX);
                int smIdx = (offsetY) * (blockDim.x + 4 * stride) + (offsetX);
                sharedFrameBuffer[smIdx] = inFrameBuffer.buffer[globalIdx];
                sharedGBuffer[smIdx] = gBuffer.buffer[globalIdx];
            }
        }
        __syncthreads();
        GBufferPixel gbp0 = sharedGBuffer[threadIdx.y * (blockDim.x + 4 * stride) + threadIdx.x];
#else // DENOISE_WITH_SHARED_MEMORY
        GBufferPixel gbp0 = gBuffer.getPixelByHW(idxY, idxX);
#endif // DENOISE_WITH_SHARED_MEMORY
        for (int dy = -2; dy <= 2; ++dy) {
            int y = idxY + dy * stride;
            if (y < 0 || y >= sizeY) {
                continue;
            }
            for (int dx = -2; dx <= 2; ++dx) {
                int x = idxX + dx * stride;
                if (x < 0 || x >= sizeX) {
                    continue;
                }

#if DENOISE_WITH_SHARED_MEMORY
                int smY = threadIdx.y + (dy + 2) * stride;
                int smX = threadIdx.x + (dx + 2) * stride;
                int smIdx = smY * (blockDim.x + 4 * stride) + smX;
                GBufferPixel gbp1 = sharedGBuffer[smIdx];
#else // DENOISE_WITH_SHARED_MEMORY
                GBufferPixel gbp1 = gBuffer.getPixelByHW(y, x);
#endif // DENOISE_WITH_SHARED_MEMORY

                float weight = denoiseKernel[dy + 2][dx + 2];

                float logColorWeight = getLogColorWeight(gbp0.baseColor, gbp1.baseColor, wParam.colorWeight * wParam.colorWeight);
                float logNormalWeight = getLogNormalWeight(gbp0.surfaceNormal, gbp1.surfaceNormal, wParam.normalWeight * wParam.normalWeight);
                float logPositionWeight = getLogPositionWeight(gbp0.position, gbp1.position, wParam.positionWeight * wParam.positionWeight);
                float logPlaneWeight = getLogPlaneWeight(gbp0.position, gbp1.position, gbp0.surfaceNormal, wParam.planeWeight * wParam.planeWeight);

                if (y != idxY && x != idxX && (wParam.colorWeight == 0.f || wParam.normalWeight == 0.f || wParam.positionWeight == 0.f || wParam.planeWeight == 0.f || gbp0.geometryId < 0)) {
                    continue;
                }

                weight *= __expf(logColorWeight + logNormalWeight + logPositionWeight + logPlaneWeight);

                totalWeight += weight;
#if DENOISE_WITH_SHARED_MEMORY
                color += sharedFrameBuffer[smIdx] * weight;
#else // DENOISE_WITH_SHARED_MEMORY
                color += inFrameBuffer.getPixelByHW(y, x) * weight;
#endif // DENOISE_WITH_SHARED_MEMORY
            }
        }
    }
    outFrameBuffer.setPixelByHW(idxY, idxX, totalWeight == 0.f ? glm::vec3(0.f) : color / totalWeight);
}

void denoise_A_Trous_EdgeAvoidingMoreParam(const dim3 blocksPerGrid2d, const dim3 blockSize2d, Texture2D<glm::vec3> outFrameBuffer, Texture2D<glm::vec3> inFrameBuffer, Texture2D<GBufferPixel> gBuffer, DenoiseWeightParam wParam) {
    cudaMemcpy(dev_tmpBuffer, inFrameBuffer.buffer, sizeof(glm::vec3) * outFrameBuffer.size.x * outFrameBuffer.size.y, cudaMemcpyDeviceToDevice);
    Texture2D<glm::vec3> tmpBuffer;
    tmpBuffer.buffer = dev_tmpBuffer;
    tmpBuffer.size = inFrameBuffer.size;

    if (wParam.filterSize == 0) {
        cudaMemcpy(outFrameBuffer.buffer, inFrameBuffer.buffer, sizeof(glm::vec3) * outFrameBuffer.size.x * outFrameBuffer.size.y, cudaMemcpyDeviceToDevice);
    }

    for (int stride = 1; stride <= wParam.filterSize; stride <<= 1) {
        if (stride > 1) {
            cudaMemcpy(dev_tmpBuffer, outFrameBuffer.buffer, sizeof(glm::vec3) * outFrameBuffer.size.x * outFrameBuffer.size.y, cudaMemcpyDeviceToDevice);
        }
#if DENOISE_WITH_SHARED_MEMORY
        denoise_A_Torus_EdgeAvoidingMoreParam_OneSweep<<<blocksPerGrid2d, blockSize2d, (sizeof(glm::vec3) + sizeof(GBufferPixel)) * (blockSize2d.x + 4 * stride) * (blockSize2d.y + 4 * stride)>>>(outFrameBuffer, tmpBuffer, gBuffer, wParam, stride);
#else // DENOISE_WITH_SHARED_MEMORY
        denoise_A_Torus_EdgeAvoidingMoreParam_OneSweep<<<blocksPerGrid2d, blockSize2d>>>(outFrameBuffer, tmpBuffer, gBuffer, wParam, stride);
#endif // DENOISE_WITH_SHARED_MEMORY
        wParam.doHalf();
    }
}

__global__ void reprojection(glm::vec3* outMisc, ui8* outValid, const glm::vec3* inLastFrameBuffer, const Texture2D<GBufferPixel> gBuffer, const GBufferPixel* lastGBuffer,
    const glm::mat4 preWorldToScreen) {
    int idxX = blockDim.x * blockIdx.x + threadIdx.x;
    int idxY = blockDim.y * blockIdx.y + threadIdx.y;

    int sizeX = gBuffer.size.x, sizeY = gBuffer.size.y;

    if (idxX < sizeX && idxY < sizeY) {
        int bufferIdx = Texture2D<glm::vec3>::index2Dto1D(gBuffer.size, idxY, idxX);
        GBufferPixel gbp0 = gBuffer.buffer[bufferIdx];
        if (gbp0.geometryId >= 0) {
            // Ignore object moving
            glm::mat4 totalMatrix = preWorldToScreen;
            glm::vec3 curPosition = gbp0.position;
            glm::vec4 preScreenWithW = totalMatrix * glm::vec4(curPosition, 1.f);
            if (preScreenWithW.w != 0.f) {
                //glm::vec3 preScreen = glm::vec3(preScreenWithW) / (preScreenWithW.w);
                //int x1 = preScreen.x + sizeX * 0.5f, y1 = preScreen.y + sizeY * 0.5f;
                int x1 = //(preScreen.x * 0.5f + 0.5f) * sizeX;
                    (static_cast<int>(preScreenWithW.x * sizeX / preScreenWithW.w) + sizeX) / 2;
                int y1 = //(preScreen.y * 0.5f + 0.5f) * sizeY;
                    (static_cast<int>(preScreenWithW.y * sizeY / preScreenWithW.w) + sizeY) / 2;
                //if(idxX == 500 && idxY == 500)
                //printf("pre[%d, %d] <%f, %f, %f> = [%f, %f, %f, %f] => [%f, %f, %f], [%d, %d]\n", 
                //    idxX, idxY, 
                //    curPosition.x, curPosition.y, curPosition.z, 
                //    preScreenWithW.x, preScreenWithW.y, preScreenWithW.z, preScreenWithW.w,
                //    preScreenWithW.x / preScreenWithW.w, preScreenWithW.y / preScreenWithW.w, preScreenWithW.z / preScreenWithW.w,
                //    x1, y1);

                // So much error?
                float absW = abs(preScreenWithW.w);
                //float sqrW = preScreenWithW.w * preScreenWithW.w;
                //sqrW *= sqrW;
                //if (abs(x1 - idxX) < sizeX * 0.03125f && abs(y1 - idxY) < sizeY * 0.03125f) {
                if (abs(x1 - idxX) < sizeX * 0.03125f * 0.25f * absW && abs(y1 - idxY) < sizeY * 0.03125f * 0.25f * absW) {
                //if (abs(x1 - idxX) * sqrW < sizeX * 100.f && abs(y1 - idxY) * sqrW < sizeY * 100.f) {
                    x1 = idxX;
                    y1 = idxY;
                }

                if (x1 >= 0 && x1 < sizeX && y1 >= 0 && y1 < sizeY) {
                    int bufferIdx1 = Texture2D<glm::vec3>::index2Dto1D(gBuffer.size, y1, x1);
                    GBufferPixel preGbp1 = lastGBuffer[bufferIdx1];
                    //glm::vec3 albedo1 = glm::vec3(glm::max(EPSILON, preGbp1.baseColor.r), glm::max(EPSILON, preGbp1.baseColor.g), glm::max(EPSILON, preGbp1.baseColor.b));
                    if (preGbp1.geometryId == gbp0.geometryId) {
                        outValid[bufferIdx] = true;
                        outMisc[bufferIdx] = inLastFrameBuffer[bufferIdx1];// / albedo1;
                    }
                    else {
                        outValid[bufferIdx] = false;
                        outMisc[bufferIdx] = glm::vec3(0.f);
                    }
                }
                else {
                    outValid[bufferIdx] = false;
                    outMisc[bufferIdx] = glm::vec3(0.f);
                }
            }
            else {
                outValid[bufferIdx] = false;
                outMisc[bufferIdx] = glm::vec3(0.f);
            }
        }
        else {
            outValid[bufferIdx] = false;
            outMisc[bufferIdx] = glm::vec3(0.f);
        }
    }
}

__global__ void accumulation(glm::vec3* inOutMisc, const ui8* inValid, const Texture2D<glm::vec3> inFrameBuffer, const DenoiseTemporalParam tParam) {
    int idxX = blockDim.x * blockIdx.x + threadIdx.x;
    int idxY = blockDim.y * blockIdx.y + threadIdx.y;

    int sizeX = inFrameBuffer.size.x, sizeY = inFrameBuffer.size.y;

    if (idxX < sizeX && idxY < sizeY) {
        int bufferIdx = Texture2D<glm::vec3>::index2Dto1D(inFrameBuffer.size, idxY, idxX);

        //// Debug
        //inOutMisc[bufferIdx] = inValid[bufferIdx] ? inOutMisc[bufferIdx] + (inFrameBuffer.buffer[bufferIdx] - inOutMisc[bufferIdx]) * tParam.alpha : glm::vec3(0.f);
        //return;

        if (inValid[bufferIdx]) {
            glm::vec3 reprojectionColor = inOutMisc[bufferIdx];
            float numFlt = 0.f;
            glm::vec3 mean(0.f), sqrMean(0.f);

            if (tParam.accumulateRadius > 0 && tParam.iter <= glm::max<int>(4, 1.f / tParam.alpha + 0.5f)) {
            //if (tParam.accumulateRadius > 0) {
                for (int y1 = glm::max(0, idxY - tParam.accumulateRadius); y1 <= idxY + tParam.accumulateRadius && y1 < sizeY; ++y1) {
                    for (int x1 = glm::max(0, idxX - tParam.accumulateRadius); x1 <= idxX + tParam.accumulateRadius && x1 < sizeX; ++x1) {
                        int bufferIdx1 = Texture2D<glm::vec3>::index2Dto1D(inFrameBuffer.size, y1, x1);
                        //if (inValid[bufferIdx1]) {
                        numFlt += 1.f;
                        glm::vec3 filteredColor1 = inFrameBuffer.buffer[bufferIdx1];
                        mean += filteredColor1;
                        sqrMean += filteredColor1 * filteredColor1;
                        //}
                    }
                }

                if (numFlt > 0.f) {
                    mean /= numFlt;
                    sqrMean /= numFlt;
                }
                glm::vec3 sqrD = sqrMean - mean * mean;
                glm::vec3 stdD = glm::sqrt(glm::max(glm::vec3(0.f), sqrD));
                reprojectionColor = glm::clamp(reprojectionColor, mean - stdD * tParam.colorBoxK, mean + stdD * tParam.colorBoxK);
            }
            inOutMisc[bufferIdx] = reprojectionColor + (inFrameBuffer.buffer[bufferIdx] - reprojectionColor) * tParam.alpha;
        }
        else {
            inOutMisc[bufferIdx] = inFrameBuffer.buffer[bufferIdx];
        }
    }
}


void denoise_temporalAccumulation(
    const dim3 blocksPerGrid2d, const dim3 blockSize2d, Texture2D<glm::vec3> inOutBuffer, Texture2D<GBufferPixel> gBuffer, 
    const Camera& camera, const Camera& lastCamera, DenoiseTemporalParam tParam) {
    reprojection<<<blocksPerGrid2d, blockSize2d>>>(dev_misc, dev_valid, dev_lastFrameBuffer, gBuffer, dev_lastGBuffer, lastCamera.worldToScreen);
    //reprojection<<<blocksPerGrid2d, blockSize2d>>>(dev_misc, dev_valid, dev_lastFrameBuffer, gBuffer, dev_lastGBuffer, lastCamera.worldToView);
    accumulation<<<blocksPerGrid2d, blockSize2d>>>(dev_misc, dev_valid, inOutBuffer, tParam);
    cudaMemcpy(inOutBuffer.buffer, dev_misc, sizeof(glm::vec3) * inOutBuffer.size.x * inOutBuffer.size.y, cudaMemcpyDeviceToDevice);
}
}