#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "../stream_compaction/common.h"

#define ERRORCHECK 1
#define STREAM_COMPACTION
#define SORT_MARTERIAL
#define CACHE_FIRST_BOUNCE
#define ANTI_ALIASING

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif
    exit(EXIT_FAILURE);
#endif
}

using StreamCompaction::Common::PerformanceTimer;
PerformanceTimer &timer()
{
    static PerformanceTimer timer;
    return timer;
}

static float totalTimeElapsed = 0.0f;

__host__ __device__
    thrust::default_random_engine
    makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4 *pbo, glm::ivec2 resolution,
                               int iter, glm::vec3 *image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

__global__ void gbufferToPBO(uchar4 *pbo, glm::ivec2 resolution, GBufferPixel *gBuffer)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        float timeToIntersect = gBuffer[index].t * 256.0;

        glm::vec3 posCol = glm::clamp(glm::abs(gBuffer[index].pos) * 30.f, 0.f, 255.f);
        glm::vec3 norCol = glm::clamp(glm::abs(gBuffer[index].nor) * 255.f, 0.f, 255.f);
        glm::vec3 col = posCol;
        pbo[index].w = 0;
        pbo[index].x = col.x;
        pbo[index].y = col.y;
        pbo[index].z = col.z;
    }
}

static Scene *hst_scene = NULL;
static glm::vec3 *dev_image = NULL;
static Geom *dev_geoms = NULL;
static Material *dev_materials = NULL;
static PathSegment *dev_paths = NULL;
static ShadeableIntersection *dev_intersections = NULL;
static GBufferPixel *dev_gBuffer = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static ShadeableIntersection *dev_intersections_cache = NULL;
static Triangle *dev_triangles = NULL;
static glm::vec4 *dev_textures = NULL;
static float *dev_kernel = NULL;
static glm::vec2 *dev_offset = NULL;
static glm::vec3 *dev_color = NULL;  // image color per pixel
static glm::vec3 *dev_color2 = NULL; // ping pong buffer

void pathtraceInit(Scene *scene)
{
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));
    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections_cache, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_textures, scene->textures.size() * sizeof(glm::vec4));
    cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(glm::vec4), cudaMemcpyHostToDevice);
    checkCUDAError("pathtraceInit");

    // denoise
    float kernel[25] = {
        0.003765,
        0.015019,
        0.023792,
        0.015019,
        0.003765,
        0.015019,
        0.059912,
        0.094907,
        0.059912,
        0.015019,
        0.023792,
        0.094907,
        0.150342,
        0.094907,
        0.023792,
        0.015019,
        0.059912,
        0.094907,
        0.059912,
        0.015019,
        0.003765,
        0.015019,
        0.023792,
        0.015019,
        0.003765,
    };
    cudaMalloc(&dev_kernel, 25 * sizeof(float));
    cudaMemcpy(dev_kernel, kernel, 25 * sizeof(float), cudaMemcpyHostToDevice);

    glm::vec2 offset[25];
    int idx = 0;
    for (int i = -2; i <= 2; i++)
    {
        for (int j = -2; j <= 2; j++)
        {
            offset[idx] = glm::vec2(i, j);
            idx++;
        }
    }
    cudaMalloc(&dev_offset, 25 * sizeof(glm::vec2));
    cudaMemcpy(dev_offset, offset, 25 * sizeof(glm::vec2), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_color, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_color, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_color2, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_color2, 0, pixelcount * sizeof(glm::vec3));
}

void pathtraceFree()
{
    cudaFree(dev_image); // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_gBuffer);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_intersections_cache);
    cudaFree(dev_triangles);
    cudaFree(dev_textures);
    cudaFree(dev_kernel);
    cudaFree(dev_offset);
    cudaFree(dev_color);
    cudaFree(dev_color2);

    checkCUDAError("pathtraceFree");
}

__global__ void generateGBuffer(
    int num_paths,
    ShadeableIntersection *shadeableIntersections,
    PathSegment *pathSegments,
    GBufferPixel *gBuffer)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection &isect = shadeableIntersections[idx];
        Ray &r = pathSegments[idx].ray;

        gBuffer[idx].t = isect.t;
        gBuffer[idx].nor = isect.surfaceNormal;
        gBuffer[idx].pos = r.origin + isect.t * r.direction;
    }
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment *pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y)
    {
        int index = x + (y * cam.resolution.x);
        PathSegment &segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

// implement antialiasing by jittering the ray
#ifdef ANTI_ALIASING
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(-0.5, 0.5);
        glm::vec2 sample = glm::vec2(u01(rng), u01(rng));
        segment.ray.direction = glm::normalize(cam.view - cam.right * cam.pixelLength.x * ((float)x + sample.x - (float)cam.resolution.x * 0.5f) - cam.up * cam.pixelLength.y * ((float)y + sample.y - (float)cam.resolution.y * 0.5f));
#else
        segment.ray.direction = glm::normalize(cam.view - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f) - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f));
#endif
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth, int num_paths, PathSegment *pathSegments, Geom *geoms, int geoms_size, ShadeableIntersection *intersections, Triangle *triangles, Material *materials, glm::vec4 *textures)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec2 uv;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec2 tmp_uv;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom &geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?
            else if (geom.type == MESH)
            {
                t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside, triangles, materials[geom.materialid], textures);
            }
            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                uv = tmp_uv;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            //The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].uv = uv;
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter, int num_paths, ShadeableIntersection *shadeableIntersections, PathSegment *pathSegments, Material *materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f)
        { // if the intersection exists...
            // Set up the RNG
            // LOOK: this is how you use thrust's RNG! Please look at
            // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f)
            {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else
            {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else
        {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 *image, PathSegment *iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

__global__ void shadeMaterial(int iter, int num_paths, ShadeableIntersection *shadeableIntersections, PathSegment *pathSegments, Material *materials, glm::vec4 *textures)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths)
    {
        return;
    }

    PathSegment &path = pathSegments[idx];
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (path.remainingBounces == 0)
    {
        return;
    }
    if (intersection.t > 0.0f)
    { // if the intersection exists...
        // Set up the RNG
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        Material material = materials[intersection.materialId];
        glm::vec3 materialColor = material.color;

        // If the material indicates that the object was a light, "light" the ray
        if (material.emittance > 0.0f)
        {
            path.remainingBounces = 0;
            path.color *= (materialColor * material.emittance);
            return;
        }
        glm::vec3 intersectionPoint = getPointOnRay(path.ray, intersection.t);
        scatterRay(path, intersectionPoint, intersection.surfaceNormal, intersection.uv, material, textures, rng);
        path.remainingBounces--;

        // If there was no intersection, color the ray black.
        // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
        // used for opacity, in which case they can indicate "no opacity".
        // This can be useful for post-processing and image compositing.
    }
    else
    {
        path.remainingBounces = 0;
        path.color = glm::vec3(0.0f);
    }
}

// https://jo.dreggn.org/home/2010_atrous.pdf
__global__ void atrous(glm::ivec2 resolution, float *kernel, glm::vec2 *offset, GBufferPixel *gBuffer, glm::vec3 *colorIn, glm::vec3 *colorOut,
                       float c_phi, float n_phi, float p_phi, float stepwidth)
{

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= resolution.x || y >= resolution.y)
    {
        return;
    }
    int index = x + (y * resolution.x);

    glm::vec3 cval = colorIn[index];
    glm::vec3 nval = gBuffer[index].nor;
    glm::vec3 pval = gBuffer[index].pos;

    glm::vec3 sum(0.f);
    float cum_w = 0.f;
    for (int i = 0; i < 25; i++)
    {
        glm::vec2 uv = glm::vec2(x, y) + offset[i] * stepwidth;
        int uvIdx = uv.x + uv.y * resolution.x;
        if (uvIdx >= 0 && uvIdx < resolution.x * resolution.y)
        {
            glm::vec3 ctmp = colorIn[uvIdx];
            glm::vec3 t = cval - ctmp;
            float dist2 = glm::dot(t, t);
            float c_w = glm::min(glm::exp(-(dist2) / c_phi), 1.f);

            glm::vec3 ntmp = gBuffer[uvIdx].nor;
            t = nval - ntmp;
            dist2 = glm::max(glm::dot(t, t) / (stepwidth * stepwidth), 0.f);
            float n_w = glm::min(glm::exp(-(dist2) / n_phi), 1.f);

            glm::vec3 ptmp = gBuffer[uvIdx].pos;
            t = pval - ptmp;
            dist2 = glm::dot(t, t);
            float p_w = glm::min(glm::exp(-(dist2) / p_phi), 1.f);
            float weight = c_w * n_w * p_w;
            sum += ctmp * weight * kernel[i];
            cum_w += weight * kernel[i];
        }
    }
    colorOut[index] = sum / cum_w;
}

// stream compaction predicate
struct path_alive
{
    __host__ __device__ bool operator()(const PathSegment &ps)
    {
        return ps.remainingBounces != 0;
    }
};

struct isect_comp
{
    __host__ __device__ bool operator()(const ShadeableIntersection &si1, const ShadeableIntersection &si2)
    {
        return si1.materialId < si2.materialId;
    }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter)
{
    timer().startGpuTimer();
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

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

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment *dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    // Empty gbuffer
    cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));

    bool iterationComplete = false;
    while (!iterationComplete && depth < traceDepth)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if defined(CACHE_FIRST_BOUNCE) && !defined(ANTI_ALIASING)
        if (depth == 0 && iter == 1)
        {
            // cache first bounce for the first iteration
            computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
                depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections, dev_triangles, dev_materials, dev_textures);
            checkCUDAError("cache first bounce");
            cudaDeviceSynchronize();
            cudaMemcpy(dev_intersections_cache, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        }
        else if (depth == 0)
        {
            // use cache of first bounce for other iterations
            cudaMemcpy(dev_intersections, dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        }
        else
        {
            computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
                depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections, dev_triangles, dev_materials, dev_textures);
            checkCUDAError("trace one bounce");
            cudaDeviceSynchronize();
        }
#else
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
            depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections, dev_triangles, dev_materials, dev_textures);
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
#endif
        if (depth == 0)
        {
            generateGBuffer<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_intersections, dev_paths, dev_gBuffer);
        }

        depth++;

#ifdef SORT_MATERIAL
        thrust::stable_sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, isect_comp());
#endif
        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.
        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_textures);

#ifdef STREAM_COMPACTION
        // stream compaction
        dev_path_end = thrust::partition(thrust::device, dev_paths, dev_path_end, path_alive());
        num_paths = dev_path_end - dev_paths;
        iterationComplete = (num_paths == 0);
#endif
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    //sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
               pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
    timer().endGpuTimer();
    float time = timer().getGpuElapsedTimeForPreviousOperation();
    totalTimeElapsed += time;
    std::cout << "   elapsed time: " << (totalTimeElapsed / iter) << "ms    | " << time << std::endl;
}

void showGBuffer(uchar4 *pbo)
{
    const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    gbufferToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, dev_gBuffer);
}

void showImage(uchar4 *pbo, int iter)
{
    const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
}

// https://jo.dreggn.org/home/2010_atrous.pdf
void denoise(int fs, float c_phi, float n_phi, float p_phi, uchar4 *pbo, int iter)
{
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    timer().startGpuTimer();

    cudaMemcpy(dev_color, dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    for (int i = 0; i < fs; i++)
    {
        int stepwidth = (1 << i) - 1;
        atrous<<<blocksPerGrid2d, blockSize2d>>>(cam.resolution, dev_kernel, dev_offset, dev_gBuffer, dev_color, dev_color2,
                                                 c_phi, n_phi, p_phi, stepwidth);

        //ping pong;
        glm::vec3 *temp = dev_color;
        dev_color = dev_color2;
        dev_color2 = dev_color;
    }
    timer().endGpuTimer();
    float time = timer().getGpuElapsedTimeForPreviousOperation();
    std::cout << time << "ms" << std::endl;

    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_color);
}