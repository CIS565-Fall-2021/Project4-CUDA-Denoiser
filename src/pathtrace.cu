#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "device_launch_parameters.h"

#define ERRORCHECK 1
#define SORT_MATERIALS 0
#define CACHE_FIRST_BOUNCE 1
#define STREAM_COMPACTION 1

#define DEPTH_OF_FIELD 0
#define ANTI_ALIASING 1

// only one of BOUNDING_BOX and OCTREE should be 1 at any give time
#define BOUNDING_BOX 1
#define OCTREE 0

// only one of these should be 1 at any give time
#define GBUFFER_T 0
#define GBUFFER_NORM 0
#define GBUFFER_POS 1

float totalTime = 0;


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

PerformanceTimer& timer()
{
    static PerformanceTimer timer;
    return timer;
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

__global__ void sendDenoiseToPBO(uchar4* pbo, glm::ivec2 resolution, glm::vec3* image) {
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
    int index = x + (y * resolution.x);
#if GBUFFER_T
    if (x < resolution.x && y < resolution.y) {

        float timeToIntersect = gBuffer[index].t * 256.0;

        pbo[index].w = 0;
        pbo[index].x = timeToIntersect;
        pbo[index].y = timeToIntersect;
        pbo[index].z = timeToIntersect;
    }
#elif GBUFFER_NORM
    if (x < resolution.x && y < resolution.y) {

        pbo[index].w = 0;
        pbo[index].x = glm::clamp(gBuffer[index].nor.x * 25.f, 0.f, 255.f);
        pbo[index].y = glm::clamp(gBuffer[index].nor.y * 25.f, 0.f, 255.f);
        pbo[index].z = glm::clamp(gBuffer[index].nor.z * 25.f, 0.f, 255.f);
    }
#elif GBUFFER_POS
    if (x < resolution.x && y < resolution.y) {

        pbo[index].w = 0;
        pbo[index].x = glm::clamp(gBuffer[index].pos.x * 25.f, 0.f, 255.f);
        pbo[index].y = glm::clamp(gBuffer[index].pos.y * 25.f, 0.f, 255.f);
        pbo[index].z = glm::clamp(gBuffer[index].pos.z * 25.f, 0.f, 255.f);
    }
#endif

}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static GBufferPixel* dev_gBuffer = NULL;
// TODO: static variables for device memory, any extra info you need, etc
static ShadeableIntersection* dev_intersections_cache = NULL;
static Triangle* dev_triangles = NULL;
static Triangle* dev_oct_triangles = NULL;
static OctreeNode* dev_octree = NULL;
static glm::vec3* dev_image1 = NULL;
static glm::vec3* dev_image2 = NULL;

void pathtraceInit(Scene *scene) {
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

    cudaMalloc(&dev_triangles, scene->mesh.triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->mesh.triangles.data(), scene->mesh.triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_oct_triangles, scene->octTriangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_oct_triangles, scene->octTriangles.data(), scene->octTriangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_octree, scene->octree.size() * sizeof(OctreeNode));
    cudaMemcpy(dev_octree, scene->octree.data(), scene->octree.size() * sizeof(OctreeNode), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));
    cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));

    cudaMalloc(&dev_image1, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image1, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_image2, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image2, 0, pixelcount * sizeof(glm::vec3));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    cudaFree(dev_gBuffer);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_intersections_cache);
    cudaFree(dev_triangles);
    cudaFree(dev_oct_triangles);
    cudaFree(dev_octree);
    cudaFree(dev_image1);
    cudaFree(dev_image2);

    checkCUDAError("pathtraceFree");
}

/**
* Function to map a random point to a sample on a unit disk. Based off of PBRT 13.6.2
* https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#ConcentricSampleDisk
*/
__host__ __device__ glm::vec2 concentricSampleDisk(glm::vec2 u) {
    // Map uniform random numbers from input to -1 to 1 range
    glm::vec2 uOffset = 2.f * u - glm::vec2(1.0f, 1.0f);

    // Handle degeneracy at origin
    if (uOffset.x == 0.0f && uOffset.y == 0.0f) {
        return glm::vec2(0.0f, 0.0f);
    }

    // Apply concentric mapping to point
    float theta, r;
    if (glm::abs(uOffset.x) > glm::abs(uOffset.y)) {
        r = uOffset.x;
        theta = (PI / 4.0f) * (uOffset.y / uOffset.x);
    }
    else {
        r = uOffset.y;
        theta = (PI / 2.0f) - (PI / 4.0f) * (uOffset.x / uOffset.y);
    }

    return r * glm::vec2(glm::cos(theta), glm::sin(theta));
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> n11u(-1, 1);

#if ANTI_ALIASING && !CACHE_FIRST_BOUNCE
        float xOffset = n11u(rng);
        float yOffset = n11u(rng);
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + xOffset)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + yOffset)
        );
#else
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );
#endif

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );

#if DEPTH_OF_FIELD
        // adapted from PBRT 6.2.3 https://pbr-book.org/3ed-2018/Camera_Models/Projective_Camera_Models
        float lensRadius = cam.lensRadius;
        float focalDistance = cam.focalDistance;

        if (lensRadius > 0) {
            // sample point on lens
            glm::vec2 pLens = lensRadius * concentricSampleDisk(glm::vec2(n11u(rng), n11u(rng)));

            //compute point on plane of focus
            float ft = focalDistance / glm::dot(cam.view, segment.ray.direction);
            glm::vec3 pFocus = cam.position + ft * segment.ray.direction;

            segment.ray.origin = cam.position + (cam.right * pLens.x) + (cam.up * pLens.y);
            segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
        }
#endif
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

__global__ void computeIntersections(
    int depth
    , int num_paths
    , PathSegment* pathSegments
    , Geom* geoms
    , Triangle* triangles
    , int numTriangles
    , glm::vec3 bottomLeft
    , glm::vec3 topRight
    , OctreeNode* nodes
    , int num_nodes
    , int geoms_size
    , ShadeableIntersection* intersections
)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index >= num_paths) return;

    PathSegment pathSegment = pathSegments[path_index];

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
        else if (geom.type == MESH)
        {
#if BOUNDING_BOX
            if (boundingBoxCheck(geom, pathSegment.ray, bottomLeft, topRight)) {
                t = meshIntersectionTest(geom, triangles, numTriangles, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
#elif OCTREE
            bool isLeaf;
            for (int i = 0; i < num_nodes; i++) {
                t = octreeIntersectionTest(nodes[i], geom, triangles, numTriangles, pathSegment.ray, tmp_intersect, tmp_normal, outside, isLeaf);
                if (t > 0.0f && isLeaf) {
                    break;
                }
            }
#else
            t = meshIntersectionTest(geom, triangles, numTriangles, pathSegment.ray, tmp_intersect, tmp_normal, outside);
#endif
        }

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
    }
    else
    {
        //The ray hits something
        intersections[path_index].t = t_min;
        intersections[path_index].materialId = geoms[hit_geom_index].materialid;
        intersections[path_index].surfaceNormal = normal;
    }
}

__global__ void shadeMaterials(
    int iter,
    int numPaths,
    int depth,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials
)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= numPaths) return;

    PathSegment& seg = pathSegments[index];
    ShadeableIntersection& inter = shadeableIntersections[index];

    if (inter.t > 0.0f) {
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, depth);
        thrust::uniform_real_distribution<float> u01(0, 1);

        Material& mat = materials[inter.materialId];
        glm::vec3 matColor = mat.color;

        // mat is a light so terminate
        if (mat.emittance > 0.0f) {
            seg.remainingBounces = 0;
            seg.color *= matColor * mat.emittance;
            return;
        }

        // determine new ray
        if (seg.remainingBounces > 0) {
            glm::vec3 intersect = getPointOnRay(seg.ray, inter.t);
            scatterRay(seg, intersect, inter.surfaceNormal, mat, rng);
        }
        else {
            seg.color = glm::vec3(0.0f, 0.0f, 0.0f);
        }
        seg.remainingBounces--;
    }
    else {
        seg.remainingBounces = 0;
        seg.color = glm::vec3(0.0f, 0.0f, 0.0f);
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
    gBuffer[idx].norm = shadeableIntersections[idx].surfaceNormal;
    gBuffer[idx].pos = getPointOnRay(pathSegments[idx].ray, shadeableIntersections[idx].t);
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

// Comparison operator can be defined for thrust sort like here: https://stackoverflow.com/questions/5282039/sorting-objects-with-thrust-cuda
struct compMats {
    __host__ __device__ bool operator()(const ShadeableIntersection& i1, const ShadeableIntersection& i2) {
        return i1.materialId < i2.materialId;
    }
};

struct continuePath {
    __host__ __device__ bool operator()(const PathSegment path) {
        return path.remainingBounces > 0;
    }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter) {
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

    ///////////////////////////////////////////////////////////////////////////

    timer().startGpuTimer();

    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    int total_paths = num_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    // clean gBuffer
    cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));

    // clean shading chunks
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    bool iterationComplete = false;
    while (!iterationComplete) {

        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if CACHE_FIRST_BOUNCE
#if OCTREE
        if (depth == 0 && iter == 1) { // first bounce of first iteration
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , dev_oct_triangles
                , hst_scene->octTriangles.size()
                , hst_scene->mesh.bottomLeft
                , hst_scene->mesh.topRight
                , dev_octree
                , hst_scene->octree.size()
                , hst_scene->geoms.size()
                , dev_intersections_cache);
            checkCUDAError("First iter first bounce cache error");
            cudaDeviceSynchronize();
            cudaMemcpy(dev_intersections, dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        }
        else if (depth == 0 && iter > 1) { // use the cached first bounce for all the following iterations
            cudaMemcpy(dev_intersections, dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        }
        else { // rest of the bounces can't be cached so compute
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , dev_oct_triangles
                , hst_scene->octTriangles.size()
                , hst_scene->mesh.bottomLeft
                , hst_scene->mesh.topRight
                , dev_octree
                , hst_scene->octree.size()
                , hst_scene->geoms.size()
                , dev_intersections);
            checkCUDAError("trace one bounce");
            cudaDeviceSynchronize();
        }
#else
        if (depth == 0 && iter == 1) { // first bounce of first iteration
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , dev_triangles
                , hst_scene->mesh.numTriangles
                , hst_scene->mesh.bottomLeft
                , hst_scene->mesh.topRight
                , dev_octree
                , hst_scene->octree.size()
                , hst_scene->geoms.size()
                , dev_intersections_cache);
            checkCUDAError("First iter first bounce cache error");
            cudaDeviceSynchronize();
            cudaMemcpy(dev_intersections, dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        }
        else if (depth == 0 && iter > 1) { // use the cached first bounce for all the following iterations
            cudaMemcpy(dev_intersections, dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        }
        else { // rest of the bounces can't be cached so compute
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , dev_triangles
                , hst_scene->mesh.numTriangles
                , hst_scene->mesh.bottomLeft
                , hst_scene->mesh.topRight
                , dev_octree
                , hst_scene->octree.size()
                , hst_scene->geoms.size()
                , dev_intersections);
            checkCUDAError("trace one bounce");
            cudaDeviceSynchronize();
        }
#endif

#else
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth
            , num_paths
            , dev_paths
            , dev_geoms
            , dev_triangles
            , hst_scene->mesh.numTriangles
            , hst_scene->mesh.bottomLeft
            , hst_scene->mesh.topRight
            , dev_octree
            , hst_scene->octree.size()
            , hst_scene->geoms.size()
            , dev_intersections);
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
#endif

        if (depth == 0) {
            generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > 
                (num_paths, dev_intersections, dev_paths, dev_gBuffer);
        }
        depth++;

        shadeMaterials << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            depth,
            dev_intersections,
            dev_paths,
            dev_materials
            );

#if SORT_MATERIALS
        // Sort by material
        thrust::stable_sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compMats());
#endif

#if STREAM_COMPACTION
        // Stream compaction
        dev_path_end = thrust::partition(
            thrust::device,
            dev_paths,
            dev_path_end,
            continuePath()); // moves all paths that can continue to front and returns new pointer to ending
        num_paths = dev_path_end - dev_paths;
#endif
        num_paths = dev_path_end - dev_paths;
        iterationComplete = (depth >= hst_scene->state.traceDepth) || (num_paths <= 0);

    }

    timer().endGpuTimer();
    printElapsedTime(timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    totalTime += timer().getGpuElapsedTimeForPreviousOperation();
    if (iter == 100) std::cout << "Total Time: " << totalTime << "ms" << std::endl;

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (total_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

__global__ void denoiseATrous(
    float c_phi,    // color
    float n_phi,    // normal
    float p_phi,    //postiion
    int stepWidth,
    GBufferPixel* gBuffer,
    glm::vec3* image_in,
    glm::vec3* image_out,
    glm::ivec2 camRes) {

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < camRes.x && y < camRes.y) {
        float kern[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
        glm::vec3 sum = glm::vec3(0.f, 0.f, 0.f);

        int index = x + (y * camRes.x);
        glm::vec3 cval = image_in[index];
        glm::vec3 nval = gBuffer[index].norm;
        glm::vec3 pval = gBuffer[index].pos;

        float cum_w = 0.0f;

        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
                float kernVal = kern[i + 2] * kern[j + 2];

                // Find neighbors
                glm::ivec2 uv = glm::clamp(glm::ivec2(x + (i * stepWidth), y + (j * stepWidth)),
                    glm::ivec2(0, 0),
                    camRes - glm::ivec2(1, 1));
                int uvIndex = uv.x + (uv.y * camRes.x);

                // Colors
                glm::vec3 ctmp = image_in[uvIndex];
                glm::vec3 t = cval - ctmp;
                float cdist = glm::dot(t, t);
                float c_w = glm::min(glm::exp(-cdist / c_phi), 1.0f);

                // Normals
                glm::vec3 ntmp = gBuffer[uvIndex].norm;
                t = nval - ntmp;
                float ndist = glm::max(glm::dot(t, t) / ((float)stepWidth * (float)stepWidth), 0.f);
                float n_w = glm::min(glm::exp(-ndist / n_phi), 1.0f);

                // Positions
                glm::vec3 ptmp = gBuffer[uvIndex].pos;
                t = pval - ptmp;
                float pdist = dot(t, t);
                float p_w = glm::min(glm::exp(-pdist / p_phi), 1.0f);

                float weight = c_w * n_w * p_w;
                sum += ctmp * weight * kernVal;
                cum_w += weight * kernVal;
            }
        }
        image_out[index] = sum / cum_w;
    }
}

__global__ void copyImageBuffer(glm::ivec2 camRes, int iter, glm::vec3* dest, const glm::vec3* src) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < camRes.x && y < camRes.y) {
        int index = x + (y * camRes.x);

        dest[index].x = src[index].x / iter;
        dest[index].y = src[index].y / iter;
        dest[index].z = src[index].z / iter;
    }
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

void showImage(uchar4* pbo, int iter) {
    const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
}

void showDenoised(uchar4* pbo, int iter, int filterSize, float c_phi, float n_phi, float p_phi) {
    timer().startGpuTimer();
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);


    copyImageBuffer << <blocksPerGrid2d, blockSize2d >> > (cam.resolution, iter, dev_image1, dev_image);

    int numIters = glm::ceil(glm::log2(filterSize));
    for (int i = 0; i < numIters; i++) {
        int stepWidth = 1 << i;
        denoiseATrous << <blocksPerGrid2d, blockSize2d >> > 
            (c_phi, n_phi, p_phi, stepWidth, dev_gBuffer, dev_image1, dev_image2, cam.resolution);

        std::swap(dev_image1, dev_image2); // ping-pong buffers
    }

    sendDenoiseToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_image1);

    cudaMemcpy(hst_scene->state.image.data(), dev_image1, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    timer().endGpuTimer();
    std::cout << timer().getGpuElapsedTimeForPreviousOperation() << std::endl;
    //printElapsedTime(timer().getGpuElapsedTimeForPreviousOperation(), "(Denoise CUDA Measured)");
}
