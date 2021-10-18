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
    float colorWeight = 0.45f;
    float normalWeight = 0.35f;
    float positionWeight = 0.2f;

    void doHalf() {
        colorWeight *= 0.5f;
        normalWeight *= 0.5f;
        positionWeight *= 0.5f;
    }
};

void init(glm::ivec2 size);

void deinit();

void denoise(DenoiserType type, const dim3 blocksPerGrid2d, const dim3 blockSize2d, Texture2D<glm::vec3> outFrameBuffer, Texture2D<glm::vec3> inFrameBuffer, Texture2D<GBufferPixel> gBuffer, DenoiseWeightParam param);

void denoise_A_Trous(const dim3 blocksPerGrid2d, const dim3 blockSize2d, Texture2D<glm::vec3> outFrameBuffer, Texture2D<glm::vec3> inFrameBuffer);
void denoise_A_Trous_EdgeAvoiding(const dim3 blocksPerGrid2d, const dim3 blockSize2d, Texture2D<glm::vec3> outFrameBuffer, Texture2D<glm::vec3> inFrameBuffer, Texture2D<GBufferPixel> gBuffer, DenoiseWeightParam param);
}
