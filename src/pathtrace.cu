#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <chrono>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
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

#define ERRORCHECK 1
#define INSTRUMENT 0
#define STREAM_COMPACTION 1
#define MATERIAL_SORT 0
#define FIRST_BOUNCE_CACHE 0
#define ANTI_ALIASING 1
#define DEPTH_OF_FIELD 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
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

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer, int viewChoice) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);

        if (viewChoice == 1)        // Normals
        {
            glm::ivec3 color;
            glm::vec3 nor = glm::abs(gBuffer[index].nor);

            color.x = glm::clamp((int)(nor.x * 255.0), 0, 255);
            color.y = glm::clamp((int)(nor.y * 255.0), 0, 255);
            color.z = glm::clamp((int)(nor.z * 255.0), 0, 255);

            pbo[index].w = 0;
            pbo[index].x = color.x;
            pbo[index].y = color.y;
            pbo[index].z = color.z;
        }
        else if (viewChoice == 2)   // Positions
        {
            glm::ivec3 color;
            glm::vec3 pos = glm::abs(gBuffer[index].pos);

            color.x = glm::clamp((int)(pos.x * 255.0 / 10), 0, 255);
            color.y = glm::clamp((int)(pos.y * 255.0 / 10), 0, 255);
            color.z = glm::clamp((int)(pos.z * 255.0 / 10), 0, 255);

            pbo[index].w = 0;
            pbo[index].x = color.x;
            pbo[index].y = color.y;
            pbo[index].z = color.z;
        }
        else if (viewChoice == 3)   // TTF
        {
            float timeToIntersect = gBuffer[index].t * 256.0;

            pbo[index].w = 0;
            pbo[index].x = timeToIntersect;
            pbo[index].y = timeToIntersect;
            pbo[index].z = timeToIntersect;
        }
    }
}

static Scene* hst_scene = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static GBufferPixel* dev_gBuffer = NULL;
// TODO: static variables for device memory, any extra info you need, etc
#if FIRST_BOUNCE_CACHE
static ShadeableIntersection* dev_firstBounceIntersections = NULL;
#endif

#if MESH_CULL
static Mesh* dev_meshes = NULL;
static Geom* dev_triangles = NULL;
#endif


void pathtraceInit(Scene* scene) {
    hst_scene = scene;
    const Camera& cam = hst_scene->state.camera;
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
#if FIRST_BOUNCE_CACHE
    cudaMalloc(&dev_firstBounceIntersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_firstBounceIntersections, 0, pixelcount * sizeof(ShadeableIntersection));
#endif

#if MESH_CULL
    cudaMalloc(&dev_meshes, scene->meshes.size() * sizeof(Mesh));
    cudaMemcpy(dev_meshes, scene->meshes.data(), scene->meshes.size() * sizeof(Mesh), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Geom));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Geom), cudaMemcpyHostToDevice);
#endif

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
#if FIRST_BOUNCE_CACHE
    cudaFree(dev_firstBounceIntersections);
#endif

#if MESH_CULL
    cudaFree(dev_meshes);
    cudaFree(dev_triangles);
#endif

    checkCUDAError("pathtraceFree");
}

/** 
 * Maps a random point to a sample on a unit disk
 */
__host__ __device__ glm::vec3 concentricSampleDisk(glm::vec2 u)
{
    // Map input to -1 to 1 range
    // glm::vec2 uOffset = 2.f * u - glm::vec2(1.f);
    glm::vec2 uOffset = u;

    // Handle degeneracy at origin
    if (uOffset.x == 0.f && uOffset.y == 0.f)
    {
        return glm::vec3(0.f);
    }

    // Apply concentric mapping to point
    float theta, r;
    if (glm::abs(uOffset.x) > glm::abs(uOffset.y))
    {
        r = uOffset.x;
        theta = PI / 4.f * (uOffset.y / uOffset.x);
    }
    else
    {
        r = uOffset.y;
        theta = PI / 2.f - PI / 4.f * (uOffset.x / uOffset.y);
    }

    return r * glm::vec3(glm::cos(theta), glm::sin(theta), 0);

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

        float xOffset = ((float)x - (float)cam.resolution.x * 0.5f);
        float yOffset = ((float)y - (float)cam.resolution.y * 0.5f);

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> disp(-1.f, 1.f);

#if ANTI_ALIASING
        xOffset += 0.5 * disp(rng);
        yOffset += 0.5 * disp(rng);
#endif

        // DONE: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * xOffset
            - cam.up * cam.pixelLength.y * yOffset
        );

#if DEPTH_OF_FIELD
        float lensRadius = cam.lensRadius;
        float focalDistance = cam.focalLength;
        float sample1 = disp(rng);
        float sample2 = disp(rng);
        
        glm::vec3 pLens = lensRadius * concentricSampleDisk(glm::vec2(sample1, sample2));
        float ft = focalDistance / glm::dot(cam.view, segment.ray.direction);
        
        glm::vec3 pFocus = cam.position + ft * segment.ray.direction;
        segment.ray.origin += cam.right * pLens.x + cam.up + pLens.y;
        segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);

#if ANTI_ALIASING
        glm::vec3 aaOffset = glm::vec3(0.001 * disp(rng), 0.001 * disp(rng), 0.f);
        segment.ray.direction = glm::normalize(segment.ray.direction + aaOffset);
#endif
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
    int depth
    , int num_paths
    , PathSegment* pathSegments
    , Geom* geoms
    , int geoms_size
    , ShadeableIntersection* intersections
#if MESH_CULL
    , Mesh* meshes
    , int meshes_size
    , Geom* triangles
    , int triangles_size
#endif
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
        bool outside = true;
        bool triangleHit = false;
        int triangleHitMaterialId;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms
        // If MESH_CULL is disabled, all mesh triangles will be in the geoms array
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
#if !MESH_CULL
            else if (geom.type == TRIANGLE)
            {
                t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
#endif

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

#if MESH_CULL
        // If MESH_CULL is enabled, mesh triangle checking will happen on a per mesh basis
        for (int i = 0; i < meshes_size; i++)
        {
            Mesh& mesh = meshes[i];
            float b = boxIntersectionTest(mesh.boundingBox, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            if (b > 0.0f)       // Mesh bounding box was hit by ray, so check all triangles inside
            {
                for (int j = mesh.triangleDataStartIndex; j < mesh.triangleDataStartIndex + mesh.numTriangles; j++)
                {
                    Geom& triangle = triangles[j];
                    t = triangleIntersectionTest(triangle, pathSegment.ray, tmp_intersect, tmp_normal, outside);

                    if (t > 0.0f && t_min > t)
                    {
                        t_min = t;
                        triangleHitMaterialId = triangle.materialid;
                        intersect_point = tmp_intersect;
                        normal = tmp_normal;
                        triangleHit = true;
                    }
                }
            }
        }
#endif

        if (hit_geom_index == -1 && !triangleHit)
        {
            intersections[path_index].t = -1.0f;
        }
#if MESH_CULL
        else if (triangleHit)
        {
            //The ray hits a triangle from a mesh
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = triangleHitMaterialId;
            intersections[path_index].surfaceNormal = normal;
        }
#endif
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
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

/**
 * Implementation of shader that uses BSDF algorithm
 */
__global__ void shadeMaterialBSDF(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    int depth
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_paths)
    {
        return;
    }

    PathSegment* segment = &pathSegments[index];
    if (segment->remainingBounces == 0)     // // No light was hit, so pixel is black
    {
        return;
    }

    // Check for existence of intersection
    ShadeableIntersection intersection = shadeableIntersections[index];
    if (intersection.t > 0.f)
    {
        // Set up the RNG
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, depth);
        thrust::uniform_real_distribution<float> u01(0, 1);

        Material material = materials[intersection.materialId];

        // If the material indicates that the object was a light, "light" the ray
        if (material.emittance > 0.0f)      // Light has been hit
        {
            pathSegments[index].color *= (material.color * material.emittance);
            segment->remainingBounces = 0;
        }
        else                                // Perform pseudo-lighting computation
        {
            glm::vec3 intersectionPoint = getPointOnRay(segment->ray, intersection.t);
            scatterRay(pathSegments[index], intersectionPoint, intersection.surfaceNormal, material, rng);
            segment->remainingBounces--;
        }
    }
    else
    {
        // Terminate the ray and set color to black
        segment->color = glm::vec3(0.f);
        segment->remainingBounces = 0;
    }
}

__global__ void generateGBuffer(
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    GBufferPixel* gBuffer,
    int viewChoice) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        gBuffer[idx].nor = shadeableIntersections[idx].surfaceNormal;
        gBuffer[idx].pos = getPointOnRay(pathSegments[idx].ray, shadeableIntersections[idx].t);
        gBuffer[idx].t = shadeableIntersections[idx].t;
    }
}

/**
 * Predicate for thrust::remove_if to remove rays that have reached the end of their life
 */
struct hasNoBounces
{
    __host__ __device__ bool operator()(const PathSegment &p)
    {
        return glm::length(p.color) < 0.0001f;
        // return p.remainingBounces <= 0;
        // Testing whether remaining bounces exist did not work, so instead I Had to check the color being black.
        // This does result in rays being filtered, as seen by observing the number of remaining paths in the pathtrace function.
    }
};

/**
 * Predicate for thrust::sort to sort intersections by material type
 */
struct sortByMaterialType
{
    __host__ __device__ bool operator()(const ShadeableIntersection &l, const ShadeableIntersection &r)
    {
        return l.materialId < r.materialId;
    }
};

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

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter, int viewChoice) {
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
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

#if INSTRUMENT
    auto startTime = std::chrono::high_resolution_clock::now();
    double intersectTime = 0.0;
    double shadeTime = 0.0;
#endif
    
    // Empty gbuffer
    cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));

    // clean shading chunks
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    bool iterationComplete = false;
    while (!iterationComplete) {
        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if INSTRUMENT
        auto startIntersectTime = std::chrono::high_resolution_clock::now();
#endif
#if FIRST_BOUNCE_CACHE
        if (depth == 0 && iter <= 1)
        {
            computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , hst_scene->geoms.size()
                , dev_firstBounceIntersections
#if MESH_CULL
                , dev_meshes
                , hst_scene->meshes.size()
                , dev_triangles
                , hst_scene->triangles.size()
#endif
                );
            checkCUDAError("trace first bounce");
            cudaDeviceSynchronize();
            cudaMemcpy(dev_intersections, dev_firstBounceIntersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        }
        else if (depth == 0)
        {
            cudaDeviceSynchronize();
            cudaMemcpy(dev_intersections, dev_firstBounceIntersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        }
        else if (depth > 0)
        {
            computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , hst_scene->geoms.size()
                , dev_intersections
#if MESH_CULL
                , dev_meshes
                , hst_scene->meshes.size()
                , dev_triangles
                , hst_scene->triangles.size()
#endif
                );
            checkCUDAError("trace one bounce");
            cudaDeviceSynchronize();
        }
#else
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
            depth
            , num_paths
            , dev_paths
            , dev_geoms
            , hst_scene->geoms.size()
            , dev_intersections
#if MESH_CULL
            , dev_meshes
            , hst_scene->meshes.size()
            , dev_triangles
            , hst_scene->triangles.size()
#endif
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
#endif
#if INSTRUMENT
        auto stopIntersectTime = std::chrono::high_resolution_clock::now();
#endif

        if (depth == 0) {
            generateGBuffer<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_intersections, dev_paths, dev_gBuffer, viewChoice);
        }
        
        depth++;

        // DONE:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.
#if MATERIAL_SORT
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, sortByMaterialType());
#endif

#if INSTRUMENT
        auto startShadeTime = std::chrono::high_resolution_clock::now();
#endif
        shadeMaterialBSDF<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            depth
            );
#if INSTRUMENT
        auto stopShadeTime = std::chrono::high_resolution_clock::now();
#endif

#if STREAM_COMPACTION
        // Remove terminated rays
        thrust::device_ptr<PathSegment> thrust_path_start = thrust::device_pointer_cast(dev_paths);
        thrust::device_ptr<PathSegment> thrust_path_end = thrust::device_pointer_cast(dev_path_end);
        thrust_path_end = thrust::remove_if(thrust::device, thrust_path_start, thrust_path_end, hasNoBounces());

        // Determine how many paths are remaining
        dev_paths = thrust::raw_pointer_cast(thrust_path_start);
        dev_path_end = thrust::raw_pointer_cast(thrust_path_end);
#endif
        num_paths = dev_path_end - dev_paths;
        iterationComplete = (depth >= traceDepth) || (num_paths == 0);

#if INSTRUMENT
        intersectTime += std::chrono::duration_cast<std::chrono::microseconds>(stopIntersectTime - startIntersectTime).count();
        shadeTime += std::chrono::duration_cast<std::chrono::microseconds>(stopShadeTime - startShadeTime).count();
#endif
    }
#if INSTRUMENT
    auto stopTime = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::microseconds>(stopTime - startTime).count();

    std::cout << totalTime << "," << intersectTime << "," << shadeTime << std::endl;
#endif

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

void showGBuffer(uchar4* pbo, int viewChoice) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    gbufferToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, dev_gBuffer, viewChoice);
}

void showImage(uchar4* pbo, int iter) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
}
