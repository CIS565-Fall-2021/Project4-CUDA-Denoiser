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
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image) {
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

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        float timeToIntersect = gBuffer[index].t * 256.0 / 10;

	glm::vec3 pn = 256.0f * gBuffer[index].p;
	pn /= 10;

	pbo[index].w = 0;
//	pbo[index].x = timeToIntersect;
//	pbo[index].y = timeToIntersect;
//	pbo[index].z = timeToIntersect;
	pbo[index].x = pn.x;
	pbo[index].y = pn.y;
	pbo[index].z = pn.z;
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static GBufferPixel* dev_gBuffer = NULL;
// TODO: static variables for device memory, any extra info you need, etc
static glm::vec3 *dev_ctmp = NULL;
__device__ float c_var = NULL;
__device__ float n_var = NULL;
__device__ float p_var = NULL;

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

	cudaMalloc(&dev_ctmp, pixelcount * sizeof(*dev_ctmp));
	
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
	cudaFree(dev_ctmp);

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

__global__ void computeIntersections(int depth, int num_paths, PathSegment * pathSegments, Geom * geoms, int geoms_size, ShadeableIntersection * intersections)
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
			Geom &geom = geoms[i];

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
			intersections[path_index].intersect_point = intersect_point;
		}
	}
}

__global__ void shadeSimpleMaterials (
	int iter,
	int num_paths,
	ShadeableIntersection *shadeableIntersections,
	PathSegment *pathSegments,
	Material *materials
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
	GBufferPixel* gBuffer)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths) {
		gBuffer[idx].t = shadeableIntersections[idx].t;
		gBuffer[idx].n = shadeableIntersections[idx].surfaceNormal;
		gBuffer[idx].p = shadeableIntersections[idx].intersect_point;
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
void pathtrace(int frame, int iter, bool denoise)
{
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
			depth,
			num_paths,
			dev_paths,
			dev_geoms,
			hst_scene->geoms.size(),
			dev_intersections
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
	
		if (depth == 0) {
			generateGBuffer<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_intersections, dev_paths, dev_gBuffer);
		}
	
		depth++;
	
		// normal and pos vectors are in dev_gBuffer at this point
		// calculate weights and variances for them (TODO)
		// need color vector


		shadeSimpleMaterials<<<numblocksPathSegmentTracing, blockSize1d>>> (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials
		);
		iterationComplete = depth == traceDepth;
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// CHECKITOUT: use dev_image as reference if you want to implement saving denoised images.
	// Otherwise, screenshots are also acceptable.
	// Retrieve image from GPU

	cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}


__constant__ float k[25] = { // gaussian blur kernel
		1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0,
		4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
		6.0 / 256.0, 24.0 / 256.0, 36.0 / 256.0, 24.0 / 256.0, 6.0 / 256.0,
		4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
		1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0
};

__constant__ struct {
	int x, y;
} offset[25] = { // offset vec2 for gaussian blur kernel
		{-2, -2}, {-2, -1}, {-2, 0}, {-2, 1}, {-2, 2},
		{-1, -2}, {-1, -1}, {-1, 0}, {-1, 1}, {-1, 2},
		{0, -2}, {0, -1}, {0, 0}, {0, 1}, {0, 2},
		{1, -2}, {1, -1}, {1, 0}, {1, 1}, {1, 2},
		{2, -2}, {2, -1}, {2, 0}, {2, 1}, {2, 2}
};

__global__ void a_trous_iter(glm::vec3 *c, GBufferPixel *gBuffer, glm::vec3* cout, int stepwidth, int resolution_x, int resolution_y, float c_weight, float n_weight, float p_weight)
{
	glm::vec3 sum;
	float w_sum = 0;

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int p = x + (y * resolution_x);
	if (x >= resolution_x || y >= resolution_y)
		return;

	for (int i = 0; i < 25; i++) {
		int q = p + (offset[i].x * stepwidth) + ((offset[i].y * resolution_x) * stepwidth);
		if (q < 0 || q > resolution_x * resolution_y)
			continue;
		
		
		glm::vec3 t = c[q] - c[p];
		float w_rt = glm::min(1.0f, glm::exp(-glm::dot(t, t) / c_var));

		t = gBuffer[q].n - gBuffer[p].n;
		float w_n = glm::min(1.0f, glm::exp(-glm::min(1.0f, (glm::dot(t, t) / (stepwidth * stepwidth))) / c_var));

		t = gBuffer[q].p - gBuffer[p].p;
		float w_p = glm::min(1.0f, glm::exp(-glm::dot(t, t) / c_var));
	
		float w = glm::pow(w_rt, c_weight) * glm::pow(w_n, n_weight) * glm::pow(w_p, p_weight);
		
		sum += c[q] * w * k[i];
		w_sum += w * k[i];
	}
	cout[p] = w_sum ? sum / w_sum : c[p];
}

/* calculate variances on GPU to avoid data transfer, an optimization is parallizing this */
__global__ void generate_variances(int num, glm::vec3 *image, GBufferPixel *gBuffer)
{
	if (blockIdx.x != 0 || threadIdx.x != 0 || blockIdx.y != 0 || threadIdx.y != 0)
		return;

	float c_avg = 0.0f, n_avg = 0.0f, p_avg = 0.0f;
	float c_dev = 0.0f, n_dev = 0.0f, p_dev = 0.0f;

	for (int i = 0; i < num; i++) {
		c_avg += glm::length(image[i]);
		n_avg += glm::length(gBuffer[i].n);
		p_avg += glm::length(gBuffer[i].p);
	}
	c_avg /= num;
	n_avg /= num;
	p_avg /= num;
	
	for (int i = 0; i < num; i++) {
		float diff = (glm::length(image[i]) - c_avg);
		c_dev += diff * diff;
		
		diff = (glm::length(gBuffer[i].n) - n_avg);
		n_dev += diff * diff;
		
		diff = (glm::length(gBuffer[i].p) - p_avg);
		p_dev += diff * diff;
	}
	c_var = c_dev / num;
	n_var = n_dev / num;
	p_var = p_dev / num;

	c_var = 1.0f;
	n_var = 1.0f;
	p_var = 1.0f;
}

void denoise(int filter_size, float color_weight, float position_weight, float normal_weight)
{
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

//	printf("here\n");
	generate_variances<<<1, 1>>>(pixelcount, dev_image, dev_gBuffer);
//	printf("there\n");

	glm::vec3 *dev_cin = dev_image;
	glm::vec3 *dev_cout = dev_ctmp;

	for (int stepwidth = 1; stepwidth < filter_size / 2; stepwidth *= 2) {
		a_trous_iter<<<blocksPerGrid2d, blockSize2d>>>(dev_cin, dev_gBuffer, dev_cout, stepwidth, cam.resolution.x, cam.resolution.y, color_weight, position_weight, normal_weight);
		std::swap(dev_cin, dev_cout);
	}
	cudaMemcpy(dev_image, dev_cout, pixelcount * sizeof(*dev_image), cudaMemcpyDeviceToDevice);


	// for saving?
	cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
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
