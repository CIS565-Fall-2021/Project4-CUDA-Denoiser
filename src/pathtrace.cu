#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/device_vector.h>
#include "device_launch_parameters.h"

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

__global__ void sendImageToDenoiseBuffer(glm::ivec2 resolution, int iter, glm::vec3* image, glm::vec3* denoise) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];
        denoise[index] = pix / (float)iter;
    }
}

__global__ void sendDenoisedImageToPBO(uchar4* pbo, glm::ivec2 resolution, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        //float timeToIntersect = gBuffer[index].t * 256.0;
        //glm::ve

        //pbo[index].w = 0;
        //pbo[index].x = timeToIntersect;
        //pbo[index].y = timeToIntersect;
        //pbo[index].z = timeToIntersect;
        //glm::vec3 pix = abs(gBuffer[index].normal);
        glm::vec3 pix = (gBuffer[index].pos + 5.f) / 10.f;
        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static glm::vec3* dev_denoised_start = NULL;
static glm::vec3* dev_denoised_end = NULL;
static Geom * dev_geoms = NULL;
static Mesh* dev_meshes = NULL;
static Triangle* dev_triangles = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static PathSegment* dev_first_bounce_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static ShadeableIntersection* dev_first_bounce_intersections = NULL;
static thrust::device_ptr<PathSegment> thrust_dev_paths;
static thrust::device_ptr<ShadeableIntersection> thrust_dev_intersections;
static int* dev_materialID = NULL;
static thrust::device_ptr<int> thrust_dev_materialID;
static int* dev_materialID_duplicate = NULL;
static thrust::device_ptr<int> thrust_dev_materialID_duplicate;
static bool cached = false;

static GBufferPixel* dev_gBuffer = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
__constant__ float kernel[25] = { 0.003765,	0.015019,	0.023792,	0.015019,	0.003765,
        0.015019,	0.059912,	0.094907,	0.059912,	0.015019,
        0.023792,	0.094907,	0.150342,	0.094907,	0.023792,
        0.015019,	0.059912,	0.094907,	0.059912,	0.015019,
        0.003765,	0.015019,	0.023792,	0.015019,	0.003765 };

__constant__ int2 offset[25];

using performanceAnalysis::PerformanceTimer;
PerformanceTimer& timer()
{
    static PerformanceTimer timer;
    return timer;
}

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_denoised_start, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised_start, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_denoised_end, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised_end, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
    thrust_dev_paths = thrust::device_pointer_cast(dev_paths);

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_meshes, scene->meshes.size() * sizeof(Mesh));
    cudaMemcpy(dev_meshes, scene->meshes.data(), scene->meshes.size() * sizeof(Mesh), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
    thrust_dev_intersections = thrust::device_pointer_cast(dev_intersections);

    cudaMalloc(&dev_first_bounce_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMalloc(&dev_first_bounce_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_materialID, pixelcount * sizeof(int));
    cudaMemset(dev_materialID, 0, pixelcount * sizeof(int));
    thrust_dev_materialID = thrust::device_pointer_cast(dev_materialID);

    cudaMalloc(&dev_materialID_duplicate, pixelcount * sizeof(int));
    thrust_dev_materialID_duplicate = thrust::device_pointer_cast(dev_materialID_duplicate);

    cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_denoised_end);
    cudaFree(dev_denoised_start);
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_meshes);
    cudaFree(dev_triangles);
    cudaFree(dev_intersections);
    cudaFree(dev_gBuffer);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_first_bounce_intersections);
    cudaFree(dev_first_bounce_paths);
    cudaFree(dev_materialID);
    cudaFree(dev_materialID_duplicate);
    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, bool stochasticAA, bool depthOfField)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> u01(0, 1);

        PathSegment & segment = pathSegments[index];
        if (stochasticAA) {
            float dx = u01(rng) * cam.pixelLength.x - cam.pixelLength.x / 2;
            float dy = u01(rng) * cam.pixelLength.y - cam.pixelLength.y / 2;
            segment.ray.origin = cam.position;
            segment.ray.direction = glm::normalize(cam.view
                - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
                - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
                - cam.right * dx - cam.up * dy);
        }
        else {
            segment.ray.origin = cam.position;
            segment.ray.direction = glm::normalize(cam.view
                - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
                - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
            );
        }

        if (depthOfField) {
            float lensRadius = 0.2f; // 0.003f
            float focalDistance = 6.f;
            thrust::normal_distribution<float> n01(0, 1);
            float theta = u01(rng) * TWO_PI;
            glm::vec3 circlePerturb = lensRadius * n01(rng) * (cos(theta) * cam.right + sin(theta) * cam.up);
            glm::vec3 originalDir = segment.ray.direction;
            float ft = focalDistance / glm::dot(originalDir, cam.view);
            segment.ray.origin = cam.position + circlePerturb;
            segment.ray.direction = glm::normalize(ft * originalDir - circlePerturb);
        }

        
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray


        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

__global__ void computeIntersections(
    int depth
    , int num_paths
    , PathSegment * pathSegments
    , Geom * geoms
    , int geoms_size, int materials_size
    , Mesh* meshes, int meshes_size
    , Triangle* triangles, int triangles_size
    , ShadeableIntersection * intersections
    , int* materialIDs, bool boundingVolumeCulling
    )
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        int hit_mesh_index = -1;
        bool outside = true;
        bool temp_outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom & geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, temp_outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, temp_outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t && (t > 0.001f || temp_outside))
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                outside = temp_outside;
            }
        }
        for (int i = 0; i < meshes_size; i++) {
            Mesh& mesh = meshes[i];
            glm::vec3 transform_max = multiplyMV(mesh.transform, glm::vec4(mesh.maxXYZ, 1.f));
            glm::vec3 transform_min = multiplyMV(mesh.transform, glm::vec4(mesh.minXYZ, 1.f));
            glm::vec3 center = (transform_max + transform_min) / 2.f;
            float radiusSquared = glm::length2(transform_max - transform_min);
            if (!boundingVolumeCulling || glm::intersectRaySphere(pathSegment.ray.origin, pathSegment.ray.direction, center, radiusSquared, t)) {
                for (int j = mesh.indexStart; j < mesh.indexEnd; j++) {
                    Triangle& triangle = triangles[j];
                    glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(pathSegment.ray.origin, 1.0f));
                    glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(pathSegment.ray.direction, 0.0f)));
                    Ray rt;
                    rt.origin = ro;
                    rt.direction = rd;
                    //temp_outside = glm::dot(pathSegment.ray.direction, triangle.normal) <= 0;
                    temp_outside = glm::dot(rd, triangle.normal) <= 0;
                    if (glm::intersectRayTriangle(ro, rd, triangle.vertex1, 
                                    triangle.vertex2, triangle.vertex3, tmp_intersect)) {
                        t = tmp_intersect[2];
                        //tmp_intersect = tmp_intersect[0] * triangle.vertex1 + tmp_intersect[1] * triangle.vertex2 + (1 - tmp_intersect[0] - tmp_intersect[1]) * triangle.vertex3;
                        tmp_intersect = getPointOnRay(rt, t);
                        tmp_intersect = multiplyMV(mesh.transform, glm::vec4(tmp_intersect, 1.0f));
                        glm::vec3 triangle_normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(triangle.normal, 0.f)));
                        t = glm::length(tmp_intersect - pathSegment.ray.origin);
                        if (t_min > t && (t > 0.001f || temp_outside)) {
                            t_min = t;
                            intersect_point = tmp_intersect;
                            normal = temp_outside ? triangle_normal : -triangle_normal;
                            hit_geom_index = -1;
                            hit_mesh_index = i;
                            outside = temp_outside;
                        }
                    }
                }
            }
        }

        if (hit_geom_index == -1 && hit_mesh_index == -1)
        {
            intersections[path_index].t = -1.0f;
            intersections[path_index].materialId = materials_size;
            materialIDs[path_index] = materials_size;
        }
        else if (hit_mesh_index != -1) {
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = meshes[hit_mesh_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].intersectionPoint = intersect_point;
            intersections[path_index].outside = outside;
            materialIDs[path_index] = meshes[hit_mesh_index].materialid;
        } 
        else
        {
            //The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].intersectionPoint = intersect_point;
            intersections[path_index].outside = outside;
            materialIDs[path_index] = geoms[hit_geom_index].materialid;
        }
    }
}

__global__ void shadeSimpleMaterials (
  int iter
  , int num_paths
    , ShadeableIntersection * shadeableIntersections
    , PathSegment * pathSegments
    , Material * materials
    )
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment segment = pathSegments[idx];
    if (segment.remainingBounces == 0) {
      return;
    }

    if (intersection.t > 0.0f) { // if the intersection exists...
      segment.remainingBounces--;
      // Set up the RNG
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, segment.remainingBounces);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        segment.color *= (materialColor * material.emittance);
        segment.remainingBounces = 0;
      }
      else {
        segment.color *= materialColor;
        glm::vec3 intersectPos = intersection.t * segment.ray.direction + segment.ray.origin;
        //scatterRay(segment, intersectPos, intersection.surfaceNormal, material, rng);
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      segment.color = glm::vec3(0.0f);
      segment.remainingBounces = 0;
    }

    pathSegments[idx] = segment;
  }
}

struct not_termintated
{
        __host__ __device__
            bool operator()(const PathSegment& p)
        {
            return p.remainingBounces > 0;
        }
};


__global__ void shadeMaterial(
    int iter
    , int num_paths
    , ShadeableIntersection* shadeableIntersections
    , PathSegment* pathSegments
    , Material* materials
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) { // if the intersection exists...
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                scatterRay(pathSegments[idx], intersection.intersectionPoint, intersection.surfaceNormal, 
                    material, intersection.outside, intersection.t, rng);
                pathSegments[idx].color *= 0.75f + u01(rng) / 4;
                pathSegments[idx].remainingBounces -= 1;
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            //pathSegments[idx].color = glm::vec3(0.3f, 0.28f, 0.2f);
            pathSegments[idx].color = glm::vec3(0.f);
            pathSegments[idx].remainingBounces = 0;
        }
    }
}


__global__ void generateGBuffer (
  int num_paths,
  ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
  GBufferPixel* gBuffer) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    gBuffer[idx].t = shadeableIntersections[idx].t;
    gBuffer[idx].normal = shadeableIntersections[idx].surfaceNormal;
    gBuffer[idx].pos = shadeableIntersections[idx].intersectionPoint;
  }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

__global__ void denoise(GBufferPixel* gBuffer, glm::ivec2 resolution, int step, glm::vec3* denoiseStart, glm::vec3* denoiseEnd, 
                        float c_sigma, float n_sigma, float p_sigma) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 cval = denoiseStart[index];
        glm::vec3 nval = gBuffer[index].normal;
        glm::vec3 pval = gBuffer[index].pos;
        glm::vec3 sum = glm::vec3(0.f);
        float cumWeight = 0.f;
        for (int i = 0; i < 25; i++) {
            int x2 = glm::clamp(x + offset[i].x * step, 0, resolution.x - 1);
            int y2 = glm::clamp(y + offset[i].y * step, 0, resolution.y - 1);
            int index2 = x2 + (y2 * resolution.x);
            glm::vec3 color = denoiseStart[index2];
            glm::vec3 norm = gBuffer[index2].normal;            
            glm::vec3 pos = gBuffer[index2].pos;
            float c_w = min(exp(-glm::length2(color - cval) / c_sigma), 1.f);
            float n_w = min(exp(-glm::length2(norm - nval) / n_sigma), 1.f);
            float p_w = min(exp(-glm::length2(pos - pval) / p_sigma / 10.f), 1.f); // Need to divide by 10 cuz position not normalized. This is scene dependent.
            float weight = c_w* n_w* p_w* kernel[i];
            sum += color * weight;
            cumWeight += weight;
        }
        denoiseEnd[index] = sum / cumWeight;
    }
}

 // Recap:
 // * Initialize array of path rays (using rays that come out of the camera)
 //   * You can pass the Camera object to that kernel.
 //   * Each path ray must carry at minimum a (ray, color) pair,
 //   * where color starts as the multiplicative identity, white = (1, 1, 1).
 //   * This has already been done for you.
 // * For each depth:
 //   * Compute an intersection in the scene for each path ray.
 //     A very naive version of this has been implemented for you, but feel
 //     free to add more primitives and/or a better algorithm.
 //     Currently, intersection distance is recorded as a parametric distance,
 //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
 //     * Color is attenuated (multiplied) by reflections off of any object
 //   * TODO: Stream compact away all of the terminated paths.
 //     You may use either your implementation or `thrust::remove_if` or its
 //     cousins.
 //     * Note that you can't really use a 2D kernel launch any more - switch
 //       to 1D.
 //   * TODO: Shade the rays that intersected something or didn't bottom out.
 //     That is, color the ray by performing a color computation according
 //     to the shader, then generate a new ray to continue the ray path.
 //     We recommend just updating the ray's PathSegment in place.
 //     Note that this step may come before or after stream compaction,
 //     since some shaders you write may also cause a path to terminate.
 // * Finally, add this iteration's results to the image. This has been done
 //   for you.
void pathtrace(uchar4 *pbo, int frame, int iter, bool sortByMaterial, 
                    bool cacheFirstBounce, bool stochasticAA, bool depthOfField, bool boundingVolumeCulling) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;
    int depth = 0;
    // Empty gbuffer
    if (iter == 1) {
        cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));
    }
    
    if (cached && cacheFirstBounce && iter > 1 && !stochasticAA) {
        cudaMemcpy(dev_intersections, dev_first_bounce_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_paths, dev_first_bounce_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
    }
    else {
        cached = false;
        generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths, stochasticAA, depthOfField);
        checkCUDAError("generate camera ray");
    }
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;   //num_path keeps tracks of the number of rays
    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete) {
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        if (depth > 0 || !cached) {
            cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));         // clean shading chunks


            // tracing            
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), hst_scene->materials.size(), 
                dev_meshes, hst_scene->meshes.size(), dev_triangles, hst_scene->triangles.size(), dev_intersections, dev_materialID,
                boundingVolumeCulling);
            checkCUDAError("trace one bounce");
            cudaDeviceSynchronize();
            //timer().startGpuTimer();
            if (sortByMaterial) {
                //thrust::sort_by_key(thrust_dev_intersections, thrust_dev_intersections + num_paths, thrust_dev_paths, SICmp());
                //thrust::sort_by_key(thrust_dev_materialID, thrust_dev_materialID + num_paths, thrust::make_zip_iterator(thrust::make_tuple(thrust_dev_intersections, thrust_dev_paths)));
                thrust::copy(thrust_dev_materialID, thrust_dev_materialID + num_paths, thrust_dev_materialID_duplicate);
                thrust::sort_by_key(thrust_dev_materialID, thrust_dev_materialID + num_paths, thrust_dev_intersections);
                thrust::sort_by_key(thrust_dev_materialID_duplicate, thrust_dev_materialID_duplicate + num_paths, thrust_dev_paths);
            }
            //timer().endGpuTimer();
        }
        
        if (cacheFirstBounce && depth == 0 && !cached) {
            cudaMemcpy(dev_first_bounce_intersections, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
            cudaMemcpy(dev_first_bounce_paths, dev_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
            cached = true;
        }

        shadeMaterial <<<numblocksPathSegmentTracing, blockSize1d>>> (
        iter, num_paths, dev_intersections, dev_paths, dev_materials);
        if (iter == 1 && depth == 0) {
            generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBuffer);
        }
        cudaDeviceSynchronize();
        PathSegment* mid = thrust::raw_pointer_cast(thrust::partition(thrust_dev_paths, thrust_dev_paths + num_paths, not_termintated()));
        if (mid == dev_paths) {
            iterationComplete = true;
        }
        dev_path_end = mid;
        num_paths = dev_path_end - dev_paths;

        depth++;
     }

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // CHECKITOUT: use dev_image as reference if you want to implement saving denoised images.
    // Otherwise, screenshots are also acceptable.
    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo) {
    const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
    gbufferToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, dev_gBuffer);
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
 ///////////////////////////////////////////////////////////////////////////

void showImage(uchar4* pbo, int iter) {
const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
}



void denoisePathTrace(uchar4* pbo, int iter, int filter_size, float c_sigma, float n_sigma, float p_sigma) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
    cudaDeviceSynchronize();
    sendImageToDenoiseBuffer << <blocksPerGrid2d, blockSize2d >> > (cam.resolution, iter, dev_image, dev_denoised_start);
    int step = 1;
    int2 offsetinit[25] = { make_int2(-2, -2), make_int2(-2, -1) , make_int2(-2, 0) , make_int2(-2, 1) , make_int2(-2, 2),
                                        make_int2(-1, -2), make_int2(-1, -1) , make_int2(-1, 0) , make_int2(-1, 1) , make_int2(-1, 2),
                                        make_int2(0, -2), make_int2(0, -1) , make_int2(0, 0) , make_int2(0, 1) , make_int2(0, 2),
                                        make_int2(1, -2), make_int2(1, -1) , make_int2(1, 0) , make_int2(1, 1) , make_int2(1, 2),
                                        make_int2(2, -2), make_int2(2, -1) , make_int2(2, 0) , make_int2(2, 1) , make_int2(2, 2) };
    cudaMemcpyToSymbol(offset, offsetinit, 25 * sizeof(int2));
    while (4 * step + 1 <= filter_size) {
        cudaDeviceSynchronize();
        denoise << <blocksPerGrid2d, blockSize2d >> > (dev_gBuffer, cam.resolution, step, dev_denoised_start, dev_denoised_end, c_sigma, n_sigma, p_sigma);
        
        step *= 2;
        glm::vec3* temp;
        temp = dev_denoised_start;
        dev_denoised_start = dev_denoised_end;
        dev_denoised_end = temp;
    }
    if (step == 1) {
        return;
    }
    sendDenoisedImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_denoised_start);
}