#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

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

__global__ void filterImage(glm::ivec2 resolution,
    int iter, Denoise denoise, glm::vec3* image, GBufferPixel* gBuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::ivec2 index2D{ x, y };
        glm::vec2 res = resolution;

        glm::vec3 pixP = gBuffer[index].color;

        // TODO: apply A-Trous Filter algorithm
        glm::vec3 sum{ 0.0 };
        float k = 0.0;

        glm::vec3 boundingMin, boundingMax;
        //for (int i = 0; i < 25; i++)
        //{
        //    glm::ivec2 qIndex2D = index2D + denoise.offset[i] * denoise.stepWidth;
        //    if (qIndex2D.x >= resolution.x || qIndex2D.y >= resolution.y || qIndex2D.x < 0 || qIndex2D.y < 0)
        //        continue; // out of bounds 
        //    if (gBuffer[index].)
        //}

        for (int i = 0; i < 25; i++)
        {
            // get q index
            glm::ivec2 qIndex2D = index2D + denoise.offset[i] * denoise.stepWidth;
            if (qIndex2D.x >= resolution.x || qIndex2D.y >= resolution.y || qIndex2D.x < 0 || qIndex2D.y < 0)
                continue; // out of bounds 
            int qIndex = qIndex2D.x + (qIndex2D.y * resolution.x);

            glm::vec3 pixQ = gBuffer[qIndex].color;

            // filter
            float h_q = denoise.kernel[i];

            //float w_n = powf(glm::max(0.f, glm::dot(gBuffer[index].normal, gBuffer[qIndex].normal)), 64);

            float w_rt = glm::exp(-glm::distance(image[index] / (float)iter, image[qIndex] / (float)iter) / (denoise.colorWeight * denoise.colorWeight + 0.0001f));
            float w_n = glm::exp(-glm::distance(gBuffer[index].normal, gBuffer[qIndex].normal) / (denoise.normalWeight * denoise.normalWeight + 0.0001f));
            float w_x = glm::exp(-glm::distance(gBuffer[index].position, gBuffer[qIndex].position)/ (denoise.positionWeight * denoise.positionWeight + 0.0001f));

            float weight = w_rt * w_n * w_x;

            // summation
            sum += h_q * weight * pixQ;
            k += h_q * weight;
        }
        if (k > 0.0001)
            gBuffer[index].color = sum / k;
    }
}

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer, GBufferType type) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);

        pbo[index].w = 0;

        if (gBuffer[index].t > 0.0)
        {
            if (type == GBufferType::NORMAL)
            {
                pbo[index].x = glm::abs(gBuffer[index].normal.x * 255.f);
                pbo[index].y = glm::abs(gBuffer[index].normal.y * 255.f);
                pbo[index].z = glm::abs(gBuffer[index].normal.z * 255.f);
            }
            else if (type == GBufferType::POSITION)
            {
                pbo[index].x = glm::clamp((gBuffer[index].position.x + 5) / 10.f * 255.f, 0.f, 255.f) * .75f;
                pbo[index].y = glm::clamp((gBuffer[index].position.y + 5) / 10.f * 255.f, 0.f, 255.f) * .75f;
                pbo[index].z = glm::clamp((gBuffer[index].position.z + 5) / 10.f * 255.f, 0.f, 255.f) * .75f;
            }
            else if (type == GBufferType::COLOR)
            {
                pbo[index].x = glm::clamp((int)(gBuffer[index].color.x * 255.0), 0, 255);
                pbo[index].y = glm::clamp((int)(gBuffer[index].color.y * 255.0), 0, 255);
                pbo[index].z = glm::clamp((int)(gBuffer[index].color.z * 255.0), 0, 255);
            }
            else
            {
                pbo[index].x = 0;
                pbo[index].y = 0;
                pbo[index].z = 0;
            }
        }
    }
}

__global__ void gbufferColorToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);

        pbo[index].w = 0;

        if (gBuffer[index].t > 0.0)
        {
            pbo[index].x = gBuffer[index].color.x * 255.f;
            pbo[index].y = gBuffer[index].color.y * 255.f;
            pbo[index].z = gBuffer[index].color.z * 255.f;
        }
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static GBufferPixel* dev_gBuffer = NULL;

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

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    cudaFree(dev_gBuffer);

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
    segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections
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

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
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
        scatterRay(segment, intersectPos, intersection.surfaceNormal, material, rng);
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

__global__ void generateGBuffer (
  int num_paths,
  ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
  GBufferPixel* gBuffer) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    gBuffer[idx].t = shadeableIntersections[idx].t;
    gBuffer[idx].position = getPointOnRay(pathSegments[idx].ray, shadeableIntersections[idx].t);
    gBuffer[idx].normal = shadeableIntersections[idx].surfaceNormal;
  }
}

__global__ void transferGBufferColor(
    glm::ivec2 resolution, int iter, GBufferPixel* gBuffer, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        image[index] = gBuffer[index].color * (float)iter;
    }
}

__global__ void FillGBufferColor(
    int num_paths, int iter, GBufferPixel* gBuffer, glm::vec3* image) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        gBuffer[idx].color = image[idx] / ((float)iter);
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

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter) {
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

    // Pathtracing Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * NEW: For the first depth, generate geometry buffers (gbuffers)
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally:
    //     * if not denoising, add this iteration's results to the image
    //     * TODO: if denoising, run kernels that take both the raw pathtraced result and the gbuffer, and put the result in the "pbo" from opengl

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

  // Empty gbuffer
  cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));

	// clean shading chunks
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

  bool iterationComplete = false;
	while (!iterationComplete) {

	// tracing
	dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
	computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
		depth
		, num_paths
		, dev_paths
		, dev_geoms
		, hst_scene->geoms.size()
		, dev_intersections
		);
	checkCUDAError("trace one bounce");
	cudaDeviceSynchronize();

    if (depth == 0) {
        generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBuffer);
    }

  shadeSimpleMaterials<<<numblocksPathSegmentTracing, blockSize1d>>> (
    iter,
    num_paths,
    dev_intersections,
    dev_paths,
    dev_materials
  );

  depth++;

  iterationComplete = depth == traceDepth;
	}

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    FillGBufferColor << <numBlocksPixels, blockSize1d >> > (pixelcount, iter, dev_gBuffer, dev_image);

    // CHECKITOUT: use dev_image as reference if you want to implement saving denoised images.
    // Otherwise, screenshots are also acceptable.
    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo, GBufferType type) {
    const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
    gbufferToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, dev_gBuffer, type);
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

void showDenoisedImage(uchar4* pbo, int iter, Denoise denoise)
{
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Fill denoiser argument
    // apply gaussian kernel with sigma 1.0
    denoise.kernel[0]  = .003765f; denoise.kernel[1]  = .015019f; denoise.kernel[2]  = .023792f; denoise.kernel[3]  = .015019f; denoise.kernel[4]  = .003765f;
    denoise.kernel[5]  = .015019f; denoise.kernel[6]  = .059912f; denoise.kernel[7]  = .094907f; denoise.kernel[8]  = .059912f; denoise.kernel[9]  = .015019f;
    denoise.kernel[10] = .023792f; denoise.kernel[11] = .094907f; denoise.kernel[12] = .150342f; denoise.kernel[13] = .094907f; denoise.kernel[14] = .023792f;
    denoise.kernel[15] = .015019f; denoise.kernel[16] = .059912f; denoise.kernel[17] = .094907f; denoise.kernel[18] = .059912f; denoise.kernel[19] = .015019f;
    denoise.kernel[20] = .003765f; denoise.kernel[21] = .015019f; denoise.kernel[22] = .023792f; denoise.kernel[23] = .015019f; denoise.kernel[24] = .003765f;

    // fill offset matrix
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            denoise.offset[i * 5 + j] = glm::ivec2{ i - 2, j - 2 };

    // TODO: cannot access pointers to gpu memory
    // calculate sigmas
   /* glm::vec3 meanNorm{ 0.0 };
    glm::vec3 meanPos{ 0.0 };
    glm::vec3 meanColor{ 0.0 };
    for (int i = 0; i < cam.resolution.x * cam.resolution.y; i++)
    {
        meanColor += dev_image[i]/((float)iter);
        meanNorm += dev_gBuffer[i].normal;
        meanPos += dev_gBuffer[i].position;
    }
    meanColor /= (cam.resolution.x * cam.resolution.y);
    meanNorm /= (cam.resolution.x * cam.resolution.y);
    meanPos /= (cam.resolution.x * cam.resolution.y);

    for (int i = 0; i < cam.resolution.x * cam.resolution.y; i++)
    {
        denoise.sigma2RT = glm::length((dev_image[i] / ((float)iter) - meanColor) * (dev_image[i] / ((float)iter) - meanColor));
        denoise.sigma2N = glm::length((dev_gBuffer[i].normal - meanNorm) * (dev_gBuffer[i].normal - meanNorm));
        denoise.sigma2X = glm::length((dev_gBuffer[i].position - meanPos) * (dev_gBuffer[i].position - meanPos));
    }
    denoise.sigma2RT /= (cam.resolution.x * cam.resolution.y);
    denoise.sigma2N /= (cam.resolution.x * cam.resolution.y);
    denoise.sigma2X /= (cam.resolution.x * cam.resolution.y);*/

    // TODO: temporary; delete
 /*   denoise.sigma2RT = 1;
    denoise.sigma2N = 1;
    denoise.sigma2X = 100;*/

    for (int i = 1; i <= (denoise.kernelSize / 5); i <<= 1)
    {
        denoise.stepWidth = i;
        /*if (i != 1)
            denoise.sigma2RT = powf(2.f, (float)-i) * denoise.sigma2RT;*/
        filterImage << <blocksPerGrid2d, blockSize2d >> > 
            (cam.resolution, iter, denoise, dev_image, dev_gBuffer);
    }
    // Send results to OpenGL buffer for rendering
    transferGBufferColor << <blocksPerGrid2d, blockSize2d >> > (cam.resolution, iter, dev_gBuffer, dev_image);
    //gbufferColorToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer);
    //sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
}
