#include "denoise.h"

extern int ui_filterSize;

namespace Denoise {

__constant__ float denoiseKernel[5][5];

glm::vec3* dev_tmpBuffer = nullptr;
#if !PREGATHER_FINAL_IMAGE
glm::vec3* tmpImageTextureBuffer = nullptr;
#endif // PREGATHER_FINAL_IMAGE

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
    cudaMalloc(&tmpImageTextureBuffer, sizeof(glm::vec3) * size.x * size.y);
#endif // PREGATHER_FINAL_IMAGE
}

void deinit() {
    cudaFree(dev_tmpBuffer);
#if !PREGATHER_FINAL_IMAGE
    cudaFree(tmpImageTextureBuffer);
#endif // PREGATHER_FINAL_IMAGE
}

void denoise(DenoiserType type, const dim3 blocksPerGrid2d, const dim3 blockSize2d, Texture2D<glm::vec3> outFrameBuffer, Texture2D<glm::vec3> inFrameBuffer, Texture2D<GBufferPixel> gBuffer, DenoiseWeightParam param) {
    switch (type) {
    case DenoiserType::A_TROUS:
        denoise_A_Trous(blocksPerGrid2d, blockSize2d, outFrameBuffer, inFrameBuffer);
        break;
    case DenoiserType::A_TROUS_EDGE_AVOIDING:
        denoise_A_Trous_EdgeAvoiding(blocksPerGrid2d, blockSize2d, outFrameBuffer, inFrameBuffer, gBuffer, param);
        break;
    default:
        cudaMemcpy(outFrameBuffer.buffer, inFrameBuffer.buffer, sizeof(glm::vec3) * outFrameBuffer.size.x * outFrameBuffer.size.y, cudaMemcpyDeviceToDevice);
        break;
    }
}

__global__ void denoise_A_Torus_OneSweep(Texture2D<glm::vec3> outFrameBuffer, const Texture2D<glm::vec3> inFrameBuffer, int stride) {
    int idxX = blockDim.x * blockIdx.x + threadIdx.x;
    int idxY = blockDim.y * blockIdx.y + threadIdx.y;

    int sizeX = outFrameBuffer.size.x, sizeY = outFrameBuffer.size.y;

    float totalWeight = 0.f;
    glm::vec3 color;
    if (idxX < sizeX && idxY < sizeY) {
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
                color += inFrameBuffer.getPixelByHW(y, x) * weight;
            }
        }
    }
    outFrameBuffer.setPixelByHW(idxY, idxX, totalWeight == 0.f ? glm::vec3(0.f) : color / totalWeight);
}

void denoise_A_Trous(const dim3 blocksPerGrid2d, const dim3 blockSize2d, Texture2D<glm::vec3> outFrameBuffer, Texture2D<glm::vec3> inFrameBuffer) {
    cudaMemcpy(dev_tmpBuffer, inFrameBuffer.buffer, sizeof(glm::vec3) * outFrameBuffer.size.x * outFrameBuffer.size.y, cudaMemcpyDeviceToDevice);
    Texture2D<glm::vec3> tmpBuffer;
    tmpBuffer.buffer = dev_tmpBuffer;
    tmpBuffer.size = inFrameBuffer.size;

    if (ui_filterSize == 0) {
        cudaMemcpy(outFrameBuffer.buffer, inFrameBuffer.buffer, sizeof(glm::vec3) * outFrameBuffer.size.x * outFrameBuffer.size.y, cudaMemcpyDeviceToDevice);
    }

    for (int stride = 1; stride <= ui_filterSize; stride <<= 1) {
        if (stride > 1) {
            cudaMemcpy(dev_tmpBuffer, outFrameBuffer.buffer, sizeof(glm::vec3) * outFrameBuffer.size.x * outFrameBuffer.size.y, cudaMemcpyDeviceToDevice);
        }
        denoise_A_Torus_OneSweep<<<blocksPerGrid2d, blockSize2d>>>(outFrameBuffer, tmpBuffer, stride);
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

__global__ void denoise_A_Torus_EdgeAvoiding_OneSweep(Texture2D<glm::vec3> outFrameBuffer, const Texture2D<glm::vec3> inFrameBuffer, const Texture2D<GBufferPixel> gBuffer, DenoiseWeightParam param, int stride) {
    int idxX = blockDim.x * blockIdx.x + threadIdx.x;
    int idxY = blockDim.y * blockIdx.y + threadIdx.y;

    int sizeX = outFrameBuffer.size.x, sizeY = outFrameBuffer.size.y;

    float totalWeight = 0.f;

    glm::vec3 color;
    if (idxX < sizeX && idxY < sizeY) {
        GBufferPixel gbp0 = gBuffer.getPixelByHW(idxY, idxX);
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

                GBufferPixel gb1 = gBuffer.getPixelByHW(y, x);

                float weight = denoiseKernel[dy + 2][dx + 2];

                float logColorWeight = getLogColorWeight(gbp0.baseColor, gb1.baseColor, param.colorWeight * param.colorWeight);
                float logNormalWeight = getLogNormalWeight(gbp0.surfaceNormal, gb1.surfaceNormal, param.normalWeight * param.normalWeight);
                float logPositionWeight = getLogPositionWeight(gbp0.position, gb1.position, param.positionWeight * param.positionWeight);

                if (y != idxY && x != idxX && (param.colorWeight == 0.f || param.normalWeight == 0.f || param.positionWeight == 0.f || gbp0.geometryId < 0)) {
                    continue;
                }

                weight *= __expf(logColorWeight + logNormalWeight + logPositionWeight);

                totalWeight += weight;
                color += inFrameBuffer.getPixelByHW(y, x) * weight;
            }
        }
    }
    outFrameBuffer.setPixelByHW(idxY, idxX, totalWeight == 0.f ? glm::vec3(0.f) : color / totalWeight);
}

void denoise_A_Trous_EdgeAvoiding(const dim3 blocksPerGrid2d, const dim3 blockSize2d, Texture2D<glm::vec3> outFrameBuffer, Texture2D<glm::vec3> inFrameBuffer, Texture2D<GBufferPixel> gBuffer, DenoiseWeightParam param) {
    cudaMemcpy(dev_tmpBuffer, inFrameBuffer.buffer, sizeof(glm::vec3) * outFrameBuffer.size.x * outFrameBuffer.size.y, cudaMemcpyDeviceToDevice);
    Texture2D<glm::vec3> tmpBuffer;
    tmpBuffer.buffer = dev_tmpBuffer;
    tmpBuffer.size = inFrameBuffer.size;

    if (ui_filterSize == 0) {
        cudaMemcpy(outFrameBuffer.buffer, inFrameBuffer.buffer, sizeof(glm::vec3) * outFrameBuffer.size.x * outFrameBuffer.size.y, cudaMemcpyDeviceToDevice);
    }

    for (int stride = 1; stride <= ui_filterSize; stride <<= 1) {
        if (stride > 1) {
            cudaMemcpy(dev_tmpBuffer, outFrameBuffer.buffer, sizeof(glm::vec3) * outFrameBuffer.size.x * outFrameBuffer.size.y, cudaMemcpyDeviceToDevice);
        }
        denoise_A_Torus_EdgeAvoiding_OneSweep<<<blocksPerGrid2d, blockSize2d>>>(outFrameBuffer, tmpBuffer, gBuffer, param, stride);
        param.doHalf();
    }
}
}