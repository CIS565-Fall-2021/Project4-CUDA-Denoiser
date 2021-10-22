#pragma once

#include "scene.h"

namespace Denoise {

#if !PREGATHER_FINAL_IMAGE
extern glm::vec3* tmpImageTextureBuffer;
#endif // PREGATHER_FINAL_IMAGE
enum class DenoiserType : ui8 {
    A_TROUS,
    A_TROUS_EDGE_AVOIDING,
    A_TROUS_EDGE_AVOIDING_MORE_PARAM,

    MAX_INDEX,
};

struct DenoiseWeightParam {
    int filterSize = 4;
    float colorWeight = 0.45f;
    float normalWeight = 0.35f;
    float positionWeight = 0.2f;
    float planeWeight = 0.1f;

    void doHalf() {
        colorWeight *= 0.5f;
        normalWeight *= 0.5f;
        positionWeight *= 0.5f;
        planeWeight *= 0.5f;
    }
};

struct DenoiseTemporalParam {
    bool enable = false;
    bool init = false;
    float alpha = 0.2f;
    float colorBoxK = 1.f;
    int accumulateRadius = 3;
    int iter = 1;
};

void init(glm::ivec2 size);

void deinit();

void denoise(
    DenoiserType type, const dim3 blocksPerGrid2d, const dim3 blockSize2d, Texture2D<glm::vec3> outFrameBuffer, Texture2D<glm::vec3> inFrameBuffer, Texture2D<GBufferPixel> gBuffer, DenoiseWeightParam wParam,
    const Camera& camera, const Camera& lastCamera, DenoiseTemporalParam tParam);

void denoise_A_Trous(const dim3 blocksPerGrid2d, const dim3 blockSize2d, Texture2D<glm::vec3> outFrameBuffer, Texture2D<glm::vec3> inFrameBuffer, DenoiseWeightParam wParam);
void denoise_A_Trous_EdgeAvoiding(const dim3 blocksPerGrid2d, const dim3 blockSize2d, Texture2D<glm::vec3> outFrameBuffer, Texture2D<glm::vec3> inFrameBuffer, Texture2D<GBufferPixel> gBuffer, DenoiseWeightParam wParam);
void denoise_A_Trous_EdgeAvoidingMoreParam(const dim3 blocksPerGrid2d, const dim3 blockSize2d, Texture2D<glm::vec3> outFrameBuffer, Texture2D<glm::vec3> inFrameBuffer, Texture2D<GBufferPixel> gBuffer, DenoiseWeightParam wParam);

void denoise_temporalAccumulation(
    const dim3 blocksPerGrid2d, const dim3 blockSize2d, Texture2D<glm::vec3> inOutFrameBuffer, Texture2D<GBufferPixel> gBuffer,
    const Camera& camera, const Camera& lastCamera, DenoiseTemporalParam tParam);
}
