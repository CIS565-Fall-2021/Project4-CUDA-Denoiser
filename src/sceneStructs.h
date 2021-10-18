#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "scenestruct/material.h"
#include "scenestruct/geometry.h"

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;

    int recordDepth = -1;
};

struct ShadeableIntersection;

enum class GBufferDataType : ui8 {
    TIME,
    BASE_COLOR,
    NORMAL,
    OBJECT_ID,
    MATERIAL_ID,
    POSITION,
};

struct GBufferPixel {
    float t = 0.f;
    i32 geometryId = -1;
    i32 materialId = -1;
    i32 stencilId = -1;
    glm::vec3 baseColor;
    glm::vec3 position;
    glm::vec3 surfaceNormal;

    GLM_FUNC_QUALIFIER void copyFromIntersection(const ShadeableIntersection& intersection);
};

struct PathSegment {
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;

    GBufferPixel gBufferData;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
    __host__ __device__ bool operator<(const ShadeableIntersection& s) const {
        return materialId < s.materialId;
    }

    float t;
    glm::vec3 surfaceNormal;
    glm::vec2 uv;
    //glm::vec3 barycentric;
    int geometryId = -1;
    int materialId = -1;
    int stencilId = -1;
};

GLM_FUNC_QUALIFIER void GBufferPixel::copyFromIntersection(const ShadeableIntersection& intersection) {
    geometryId = intersection.geometryId;
    materialId = intersection.materialId;
    surfaceNormal = intersection.surfaceNormal;
    stencilId = intersection.stencilId;
    t = intersection.t;
}

//// CHECKITOUT - a simple struct for storing scene geometry information per-pixel.
//// What information might be helpful for guiding a denoising filter?
//struct GBufferPixel {
//  float t;
//};
