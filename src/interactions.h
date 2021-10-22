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

__host__ __device__
glm::vec3 calculateRandomDirectionInSphere(thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(-1.f, 1.0f);
    while (true)
    {
        glm::vec3 p = glm::vec3(u01(rng), u01(rng), u01(rng));
        float length = glm::length(p);
        if (length >= 1.0f) { continue; }
        return p;
    }
}

__host__ __device__
float reflectance(float cosine, float refIdx)
{
    // Use Schlick's approximation for reflectance.
    float r0 = (1.0f - refIdx) / (1.0f + refIdx);
    r0 *= r0;
    return r0 + (1.0f - r0) * glm::pow(1.0f - cosine, 5.0f);
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */

__host__ __device__
void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const Material& m,
    thrust::default_random_engine& rng,
    const ShadeableIntersection& sInter) {

    thrust::uniform_real_distribution<float> u01(0, 1);

    if (m.hasReflective)
    {
        pathSegment.ray.origin = intersect + EPSILON * normal;
        glm::vec3 reflectedDir = glm::reflect(glm::normalize(pathSegment.ray.direction),
            normal);
        // add fuzziness
        reflectedDir += m.fuzziness * calculateRandomDirectionInSphere(rng);

        if (glm::dot(reflectedDir, normal) > 0.0f)
        {
            pathSegment.color *= m.specular.color;
            pathSegment.ray.direction = glm::normalize(reflectedDir);
            pathSegment.remainingBounces--;
        }
        // for big sphere or grazing rays, we may scatter below the 
        // surface. In this case, terminate this segment. 
        else {
            // NOTE: this line is necessary to prevent the white boundary
            // if we terminate the ray path because reflected ray goes below
            // the surface, this path's contribution should be set to black (0.f)
            pathSegment.color *= glm::vec3(0.f);
            pathSegment.remainingBounces = 0;
        }
    }
    else if (m.hasRefractive)
    {
        float refractionRatio = outside ? (1.f / m.indexOfRefraction) : m.indexOfRefraction;
        // refractive rays always shoots towards negative normal direction: that's why we use subtraction
        // since intersect falls slightly short to the object it's hitting, 
        // we need a bigger EPSILON so that reflective rays are shoot 
        // from a point that is not occluded by the surface
        glm::vec3 unitRayDir = glm::normalize(pathSegment.ray.direction);
        float cosTheta = fmin(glm::dot(-unitRayDir, normal), 1.0f);
        float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
        bool cannotReflect = refractionRatio * sinTheta > 1.0f;
        glm::vec3 newRayDir;
        if (cannotReflect || reflectance(cosTheta, refractionRatio) > u01(rng))
        {
            pathSegment.ray.origin = intersect + EPSILON * normal;
            newRayDir = glm::reflect(unitRayDir, normal);
        }
        else {
            pathSegment.ray.origin = intersect - EPSILON * 100.0f * normal;
            newRayDir = glm::refract(unitRayDir, normal, refractionRatio);
        }
        pathSegment.color *= m.color;
        pathSegment.ray.direction = glm::normalize(newRayDir);
        pathSegment.remainingBounces--;
    }
    else {
        pathSegment.ray.origin = intersect + EPSILON * normal;
        glm::vec3 diffuseDir = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
        pathSegment.color *= m.color;
        pathSegment.ray.direction = diffuseDir;
        // diffuse always scatter
        pathSegment.remainingBounces--;
    }

}
