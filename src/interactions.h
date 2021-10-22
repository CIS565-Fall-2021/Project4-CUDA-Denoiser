#pragma once

#define EPSILON_SCALE 10
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

/**
 * Simple ray scattering with diffuse and perfect specular support.
 */
__host__ __device__
void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    thrust::uniform_real_distribution<float> u01(0, 1);
    float prob = u01(rng);
    glm::vec3 rayVec = glm::normalize(pathSegment.ray.direction);
    glm::vec3 norVec = glm::normalize(normal);

    if (prob < m.hasReflective) {
        // Reflection
        pathSegment.ray.origin = intersect + (float)EPSILON * EPSILON_SCALE * norVec;
        pathSegment.ray.direction = glm::normalize(glm::reflect(rayVec, norVec));
    }
    else if (prob < (m.hasReflective + m.hasRefractive)) {
        // Refraction 
        // Reference: https://raytracing.github.io/books/RayTracingInOneWeekend.html#dielectrics
        float refractionRatio = (glm::dot(rayVec, norVec) > 0) ? m.indexOfRefraction : (1.0f / m.indexOfRefraction); // inside sphere : outside sphere

        float cosTheta = glm::min(glm::dot(-1.0f * rayVec, norVec), 1.0f);
        float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
        bool cannotRefract = refractionRatio * sinTheta > 1.0f;

        float r0 = (1 - refractionRatio) / (1 + refractionRatio);
        r0 *= r0;
        float schlickAppro = r0 + (1 - r0) * pow(1 - cosTheta, 5);
        bool  schlickBool = schlickAppro > u01(rng);

        pathSegment.ray.origin = intersect + (float)EPSILON * EPSILON_SCALE * rayVec;
        if (cannotRefract || schlickBool) {
            pathSegment.ray.direction = glm::reflect(rayVec, norVec);
        }
        else {
            // Snell's Law
            pathSegment.ray.direction = glm::refract(rayVec, norVec, refractionRatio);
        }
    }
    else {
        // Diffusion
        pathSegment.ray.origin = intersect + (float)EPSILON * EPSILON_SCALE * norVec;
        pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(norVec, rng));
    }
    pathSegment.color *= m.color;
}