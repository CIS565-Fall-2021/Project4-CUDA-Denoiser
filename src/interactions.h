#pragma once

#include "intersections.h"

/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ float getFresnelCoefficient(float eta, float cosTheta, float matIOF) {
    // handle total internal reflection
    float sinThetaI = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
    float sinThetaT = eta * sinThetaI;
    float fresnelCoeff = 1.f;

    cosTheta = abs(cosTheta);
    if (sinThetaT < 1) {

        float cosThetaT = sqrt(max(0.f, 1.f - sinThetaT * sinThetaT));

        float rparl = ((matIOF * cosTheta) - (cosThetaT)) / ((matIOF * cosTheta) + (cosThetaT));
        float rperp = ((cosTheta)-(matIOF * cosThetaT)) / ((cosTheta)+(matIOF * cosThetaT));
        fresnelCoeff = (rparl * rparl + rperp * rperp) / 2.0;
    }
    return fresnelCoeff;
}

__host__ __device__
void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {

    glm::vec3 newDir;

    // specular surface
    if (m.hasReflective) {
        newDir = glm::reflect(pathSegment.ray.direction, normal);
    }
    else if (m.hasRefractive) {
        const glm::vec3& wi = pathSegment.ray.direction;

        float cosTheta = dot(normal, wi);

        // incoming direction should be opposite normal direction if entering medium
        bool entering = cosTheta < 0;
        glm::vec3 faceForwardN = !entering ? -normal : normal;

        // if entering, divide air iof (1.0) by the medium's iof
        float eta = entering ? 1.f / m.indexOfRefraction : m.indexOfRefraction;
        float fresnelCoeff = getFresnelCoefficient(eta, cosTheta, m.indexOfRefraction);

        thrust::uniform_real_distribution<float> u01(0, 1);
        if (u01(rng) < fresnelCoeff) {
            newDir = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
        }
        else {
            newDir = glm::normalize(glm::refract(wi, faceForwardN, eta));
        }

        pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction;
        pathSegment.ray.direction = newDir;
        return;

    }
    // diffuse surface
    else {
        newDir = calculateRandomDirectionInHemisphere(normal, rng);
    }

    pathSegment.ray.origin = intersect;
    pathSegment.ray.direction = newDir;
}
