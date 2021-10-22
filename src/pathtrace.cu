#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include <thrust/device_vector.h>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

#define SORT_MATERIALS false
#define CACHE_FIRST_BOUNCE false
#define DOF false
#define FOCAL_LEN 4.45f
#define ANTIALIASING false

#define SHOW_T 0
#define SHOW_POS 1
#define SHOW_NOR 2
#define SHOW_GBUFFER_TYPE 1

void checkCUDAErrorFn(const char* msg, const char* file, int line) {
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

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);

        if (SHOW_GBUFFER_TYPE == SHOW_T) {
            float timeToIntersect = gBuffer[index].t * 256.0;

            pbo[index].w = 0;
            pbo[index].x = timeToIntersect;
            pbo[index].y = timeToIntersect;
            pbo[index].z = timeToIntersect;
        }

        else if (SHOW_GBUFFER_TYPE == SHOW_POS) {
            glm::vec3 position = glm::normalize(abs(gBuffer[index].pos)) * glm::vec3(255.f);
            pbo[index].w = 0;
            pbo[index].x = position.x;
            pbo[index].y = position.y;
            pbo[index].z = position.z;
        }

        else {
            glm::vec3 normal = gBuffer[index].nor;
            normal = abs(glm::vec3(normal.x * 255.f, normal.y * 255.f, normal.z * 255.f));
            pbo[index].w = 1;
            pbo[index].x = normal.x;
            pbo[index].y = normal.y;
            pbo[index].z = normal.z;
        }
    }
}


__global__ void sendDenoisedToPBO(uchar4* pbo, glm::ivec2 resolution,
    int iter, glm::vec3* image) {
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

static Scene* hst_scene = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static thrust::device_ptr<PathSegment*> dev_thrust_alive_paths = NULL;
static PathSegment** dev_alive_paths = NULL;
static PathSegment* dev_first_paths = NULL;
static Triangle* dev_triangles = NULL;
static GBufferPixel* dev_gBuffer = NULL;
static float* dev_gaussian_kernel = NULL;
static float* dev_gaussian_offsets = NULL;
static glm::vec3* dev_denoised = NULL;
static glm::vec3* dev_denoised_tmp = NULL;

// TODO: static variables for device memory, any extra info you need, etc
// ...

void pathtraceInit(Scene* scene) {
    hst_scene = scene;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_first_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_alive_paths, pixelcount * sizeof(PathSegment*));
    dev_thrust_alive_paths = thrust::device_ptr<PathSegment*>(dev_alive_paths);

    for (int i = 0; i < scene->geoms.size(); i++) {
        if (scene->geoms[i].type == MESH) {
            cudaMalloc(&dev_triangles, scene->geoms[i].numTriangles * sizeof(Triangle));
            cudaMemcpy(dev_triangles, scene->geoms[i].triangles, scene->geoms[i].numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);
        }
    }
    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

    cudaMalloc(&dev_gaussian_kernel, 25 * sizeof(float));
    float kernel[25] = {
        0.003765,	0.015019,	0.023792,	0.015019,	0.003765,
        0.015019,	0.059912,	0.094907,	0.059912,	0.015019,
        0.023792,	0.094907,	0.150342,	0.094907,	0.023792,
        0.015019,	0.059912,	0.094907,	0.059912,	0.015019,
        0.003765,	0.015019,	0.023792,	0.015019,	0.003765,
    };
    cudaMemcpy(dev_gaussian_kernel, &kernel[0], 25 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_gaussian_offsets, 25 * sizeof(float));
    cudaMemset(dev_gaussian_offsets, 0, 25 * sizeof(float));

    cudaMalloc(&dev_denoised, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised, 0, pixelcount * sizeof(glm::vec3));
    cudaMalloc(&dev_denoised_tmp, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised_tmp, 0, pixelcount * sizeof(glm::vec3));

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_gaussian_kernel);
    cudaFree(dev_gaussian_offsets);
    cudaFree(dev_denoised);
    cudaFree(dev_denoised_tmp);
    // TODO: clean up any extra device memory you created

    checkCUDAError("pathtraceFree");
}

__global__ void generateGBuffer(
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    GBufferPixel* gBuffer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_paths)
    {
        ShadeableIntersection isect = shadeableIntersections[idx];
        Ray ray = pathSegments[idx].ray;
        gBuffer[idx].t = isect.t;
        gBuffer[idx].pos = ray.origin + glm::vec3(isect.t) * ray.direction;
        gBuffer[idx].nor = isect.surfaceNormal;
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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, PathSegment** aliveSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];
        aliveSegments[index] = &segment;

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, x, y);
        thrust::uniform_real_distribution<float> u01(0, 1);

        // calculate the ray origin
        if (DOF) {
            float aperture = 0.1;
            float sampleX = u01(rng);
            float sampleY = u01(rng);

            // warp pt to disk
            float r = sqrt(sampleX);
            float theta = 2 * 3.14159 * sampleY;
            glm::vec2 res = glm::vec2(cos(theta), sin(theta)) * r;

            segment.ray.origin = cam.position + glm::vec3(res.x, res.y, 0) * aperture;
        }
        else {
            segment.ray.origin = cam.position;
        }

        if (ANTIALIASING) {
            float rand1 = u01(rng);
            float rand2 = u01(rng);

            x = x + rand1 * 2.0;
            y = y + rand2 * 2.0;
        }

        // calculate the ray direction
        if (DOF) {
            float focalLen = FOCAL_LEN;
            float angle = glm::radians(cam.fov.y);
            float aspect = ((float)cam.resolution.x / (float)cam.resolution.y);
            float ndc_x = 1.f - ((float)x / cam.resolution.x) * 2.f;
            float ndc_y = 1.f - ((float)y / cam.resolution.x) * 2.f;

            glm::vec3 ref = cam.position + cam.view * focalLen;
            glm::vec3 H = tan(angle) * focalLen * cam.right * aspect;
            glm::vec3 V = tan(angle) * focalLen * cam.up;
            glm::vec3 target_pt = ref + V * ndc_y + H * ndc_x;
            segment.ray.direction = normalize(target_pt - segment.ray.origin);
        }
        else {
            segment.ray.direction = glm::normalize(cam.view
                - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
                - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
            );

        }

        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.terminated = false;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth
    , int num_paths
    , PathSegment** pathSegments
    , Geom* geoms
    , int geoms_size
    , ShadeableIntersection* intersections
    , Triangle* triangles
)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = *pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == MESH) {
                t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, triangles);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
            //pathSegment.remainingBounces = 0;
        }
        else
        {
            //The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
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
    int iter
    , int num_paths
    , ShadeableIntersection* shadeableIntersections
    , PathSegment** pathSegments
    , Material* materials
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) { // if the intersection exists...
          // Set up the RNG
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx]->remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx]->color *= materialColor * material.emittance;
                pathSegments[idx]->terminated = true;
            }
            else {
                // multiply by the albedo color
                pathSegments[idx]->color *= materialColor;

                // find and set next ray direction
                glm::vec3 intersectPt = getPointOnRay(pathSegments[idx]->ray, intersection.t);
                scatterRay(*pathSegments[idx], intersectPt, intersection.surfaceNormal, material, rng);
                pathSegments[idx]->remainingBounces -= 1;
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx]->color = glm::vec3(0.0f);
            pathSegments[idx]->terminated = true;
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

__device__ float w(GBufferPixel& p, GBufferPixel& q, glm::vec3 colP, glm::vec3 colQ, float colW, float norW, float posW) {
    float w_rt = min(exp(-dot(colP - colQ, colP - colQ) / colW), 1.0);
    float w_n = exp(-dot(p.nor - q.nor, p.nor - q.nor) / norW);
    float w_x = exp(-dot(p.pos - q.pos, p.pos - q.pos) / posW);
    return w_rt * w_n * w_x;
}

__global__ void applyATrousFilter(int nPaths, glm::vec3* dst, glm::vec3* prev_iter, glm::vec3* beauty,
    float resolution, float* kernel, float offset, GBufferPixel* gbuffers,
    float colW, float norW, float posW, int numIter, bool firstIter)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        glm::vec3 cur_color = glm::vec3(0);
        float k = 0;

        for (int x = -2; x < 3; ++x) {
            for (int y = -2; y < 3; ++y) {
                int newX = x * offset;
                int newY = y * offset * resolution;
                int l_index = index + newX + newY;

                if (!(l_index < 0 || l_index >= nPaths)) {
                    float h = kernel[(x + 2) + 5 * (y + 2)];
                    glm::vec3 cq = firstIter ? beauty[l_index] / (float)numIter : dst[l_index];
                    glm::vec3 cp = firstIter ? beauty[index] / (float)numIter : dst[index];
                    float wW = w(gbuffers[index], gbuffers[l_index], cp, cq, colW, norW, posW);
                    cur_color += cq * h * wW;
                    k += h * wW;
                }

            }
        }
        dst[index] += (cur_color / k) - prev_iter[index];
        prev_iter[index] = cur_color / k;
    }
}

void denoiseImage(float filterSize, float colW, float norW, float posW, int numIter) {

    std::clock_t start2;
    double duration2;
    start2 = std::clock();
    //std::cout << "starting clock at: " << start2 << std::endl;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    const int blockSize1d = 128;
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;

    cudaMemset(dev_denoised_tmp, 0, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised, 0, pixelcount * sizeof(glm::vec3));

    float iterations = filterSize < 5 ? 0 : floor(log2(filterSize / 5.f));
    float offset = 1;

    for (int i = 0; i < iterations; ++i) {
        offset = pow(2, i);
        if (i == 0) {
            applyATrousFilter << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_denoised, dev_denoised_tmp, dev_image, cam.resolution.x,
                dev_gaussian_kernel, offset, dev_gBuffer, colW, norW, posW, numIter, true);
        }
        else {
            applyATrousFilter << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_denoised, dev_denoised_tmp, dev_image, cam.resolution.x,
                dev_gaussian_kernel, offset, dev_gBuffer, colW, norW, posW, numIter, false);
        }
    }

    // duration2 = (std::clock() - start2) / (double)CLOCKS_PER_SEC;

     //cout.precision(17);
     //std::cout << "ending clock at: " << std::clock() << std::endl;
     //std::cout << "denoise time: " << fixed << duration2 << '\n';
}

// terminates ray if its terminated flag is raised
struct terminateRay {
    __host__ __device__ bool operator()(const PathSegment* ps) {
        return !ps->terminated;
    }
};

// compares materials for sorting
struct compMaterialID : public binary_function<ShadeableIntersection, ShadeableIntersection, bool> {
    __host__ __device__ bool operator()(const ShadeableIntersection& isect1, const ShadeableIntersection& isect2) {
        return isect1.materialId < isect2.materialId;
    }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter, bool denoise) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    bool isFirstIter = iter == 1;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths, dev_alive_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int init_num_paths = dev_path_end - dev_paths;
    int num_paths = init_num_paths;

    bool iterationComplete = false;
    thrust::device_ptr<PathSegment*> endPtr(dev_alive_paths + pixelcount);

    // if not the first iteration, assume the paths have been cached, harvest
    if (CACHE_FIRST_BOUNCE && !ANTIALIASING && !DOF && !isFirstIter) {
        cudaMemcpy(dev_paths, dev_first_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
        depth++; // start on second bounce now
    }

    while (!iterationComplete) {

        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth
            , num_paths
            , dev_alive_paths
            , dev_geoms
            , hst_scene->geoms.size()
            , dev_intersections
            , dev_triangles
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        if (depth == 0) {
            generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBuffer);
        }

        depth++;

        // sort rays by material
        if (SORT_MATERIALS) {
            thrust::device_ptr<PathSegment*> sorted_paths(dev_alive_paths);
            thrust::device_ptr<ShadeableIntersection> sorted_isects(dev_intersections);
            thrust::sort_by_key(sorted_isects, sorted_isects + num_paths, sorted_paths, compMaterialID());
        }

        // shade paths
        shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_alive_paths,
            dev_materials
            );

        // if first iteration, cache first bounce
        if (CACHE_FIRST_BOUNCE && !ANTIALIASING && !DOF && isFirstIter && depth == 1) {
            cudaMemcpy(dev_first_paths, dev_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
        }

        // perform stream compaction
        thrust::device_ptr<PathSegment*> newPathsEnd = thrust::partition(dev_thrust_alive_paths, endPtr, terminateRay());
        endPtr = newPathsEnd;
        num_paths = endPtr - dev_thrust_alive_paths;

        // if reached max depth or if no paths remain, terminate iteration
        if (depth == traceDepth || num_paths == 0) {
            iterationComplete = true;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (init_num_paths, dev_image, dev_paths);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
    gbufferToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer);
}

void showImage(uchar4* pbo, int iter, bool denoise) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    if (denoise) {
        sendDenoisedToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_denoised);
    }
    else {
        sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
    }
}