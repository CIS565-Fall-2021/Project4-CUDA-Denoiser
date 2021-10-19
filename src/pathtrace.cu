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

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    //if (x < resolution.x && y < resolution.y) {
    //    int index = x + (y * resolution.x);
    //    float timeToIntersect = gBuffer[index].t * 256.0;

    //    pbo[index].w = 0;
    //    pbo[index].x = timeToIntersect;
    //    pbo[index].y = timeToIntersect;
    //    pbo[index].z = timeToIntersect;
    //}

    // normal
    //if (x < resolution.x && y < resolution.y) {
    //    int index = x + (y * resolution.x);
    //    glm::vec3 normal2Color = gBuffer[index].normal * float(256.0);

    //    pbo[index].w = 0;
    //    pbo[index].x = normal2Color.x;
    //    pbo[index].y = normal2Color.y;
    //    pbo[index].z = normal2Color.z;
    //}

    // position
    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 position2Color = gBuffer[index].position * float(256.0);

        pbo[index].w = 0;
        pbo[index].x = position2Color.x;
        pbo[index].y = position2Color.y;
        pbo[index].z = position2Color.z;
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static GBufferPixel* dev_gBuffer = NULL;

static glm::vec2* dev_offsets = NULL;
static float* dev_filter = NULL;

static glm::vec3* dev_atrous_in = NULL;
static glm::vec3* dev_atrous_out = NULL;

glm::vec2* generateOffsets() {
    glm::vec2 offsets[25];

    int offsets_index = 0;

    for (int i = -2; i < 3; i++) {
        for (int j = -2; j < 3; j++) {
            offsets[offsets_index++] = glm::vec2(i, j);
        }
    }

    return offsets;
}

void denoiseInit(Scene *scene) {
    hst_scene = scene;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    float filter[25] = { 0.003765, 0.015019, 0.023792, 0.015019, 0.003765,
                    0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
                    0.023792, 0.094907, 0.150342, 0.094907, 0.023792,
                    0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
                    0.003765, 0.015019, 0.023792, 0.015019, 0.003765 };
    cudaMalloc(&dev_filter, 25 * sizeof(float));
    cudaMemcpy(dev_filter, filter, 25 * sizeof(float), cudaMemcpyHostToDevice);

    glm::vec2* offsets = generateOffsets();
    cudaMalloc(&dev_offsets, 25 * sizeof(glm::vec2));
    cudaMemcpy(dev_offsets, offsets, 25 * sizeof(glm::vec2), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_atrous_in, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_atrous_in, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_atrous_out, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_atrous_out, 0, pixelcount * sizeof(glm::vec3));

    checkCUDAError("denoiseInit");
    cudaDeviceSynchronize();
}

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

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
    cudaDeviceSynchronize();
}

void denoiseFree() {
    cudaFree(dev_filter);
    cudaFree(dev_offsets);
    cudaFree(dev_atrous_in);
    cudaFree(dev_atrous_out);

    checkCUDAError("denoiseFree");
    cudaDeviceSynchronize();
}

void pathtraceFree() {
    // no-op if dev_image is null
    cudaFree(dev_image);
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    cudaFree(dev_gBuffer);

    checkCUDAError("pathtraceFree");
    cudaDeviceSynchronize();
}

__global__ void generateRayFromCamera(
    Camera cam,
    int iter, int traceDepth,
    PathSegment* pathSegments
) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f));

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
	}
}

__global__ void computeIntersections(
    int depth, int num_paths,
    PathSegment * pathSegments,
    Geom * geoms, int geoms_size,
    ShadeableIntersection * intersections
) {
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths) {
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

		for (int i = 0; i < geoms_size; i++) {
			Geom & geom = geoms[i];

			if (geom.type == CUBE) {
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			} else if (geom.type == SPHERE) {
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t) {
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1) {
			intersections[path_index].t = -1.0f;
		} else {
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}

__global__ void shadeSimpleMaterials (
    int iter, int num_paths,
    ShadeableIntersection * shadeableIntersections,
    PathSegment * pathSegments, Material * materials
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_paths) {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        PathSegment segment = pathSegments[idx];

        if (segment.remainingBounces == 0) {
            return;
        }

        if (intersection.t > 0.0f) {
            segment.remainingBounces--;
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, segment.remainingBounces);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                segment.color *= (materialColor * material.emittance);
                segment.remainingBounces = 0;
            } else {
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
    GBufferPixel* gBuffer
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_paths) {
        gBuffer[idx].t = shadeableIntersections[idx].t;
        gBuffer[idx].normal = shadeableIntersections[idx].surfaceNormal;
        gBuffer[idx].position = getPointOnRay(pathSegments[idx].ray, shadeableIntersections[idx].t);
    }
}

__global__ void kernDenoise(
    int resX, int resY,
    GBufferPixel* gBuffer,
    glm::vec3* image,
    float* kernel,
    glm::vec2* offset,
    float stepWidth,
    float c_phi, float n_phi, float p_phi,
    glm::vec3* atrous_in, glm::vec3* atrous_out
) {
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int pixelcount = resX * resY;
    
    if (index < pixelcount) {
        const int pixelY = int(index / resX);
        const int pixelX = index - (pixelY * resX);
        glm::vec2 pixelCoord = glm::vec2(pixelX, pixelY);

        glm::vec3 cval = atrous_in[index] / 255.f;
        glm::vec3 nval = gBuffer[index].normal;
        glm::vec3 pval = gBuffer[index].position;

        glm::vec3 sum;
        float cumulative_w = 0.0;

        for (int i = 0; i < 25; i++) {
            glm::vec2 uv = pixelCoord + offset[i] * stepWidth;
            int uvIndex = uv.x + uv.y * resX;

            if (0 <= uvIndex && uvIndex < pixelcount) {
                glm::vec3 ctmp = atrous_in[uvIndex] / 255.f;
                glm::vec3 t = cval - ctmp;
                float dist2 = glm::dot(t, t);
                float c_w = min(exp(-dist2 / c_phi), float(1.0));

                glm::vec3 ntmp = gBuffer[uvIndex].normal;
                t = nval - ntmp;
                dist2 = max(glm::dot(t, t) / (stepWidth * stepWidth), float(0.0));
                float n_w = min(exp(-dist2 / n_phi), float(1.0));

                glm::vec3 ptmp = gBuffer[uvIndex].position;
                t = pval - ptmp;
                dist2 = glm::dot(t, t);
                float p_w = min(exp(-dist2 / p_phi), float(1.0));

                float weight = c_w * n_w * p_w;
                sum += ctmp * weight * kernel[i];
                cumulative_w += weight * kernel[i];
            }
        }

        atrous_out[index] = sum / cumulative_w * 255.f;
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths) {
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
		    depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections);
	    checkCUDAError("trace one bounce");
	    cudaDeviceSynchronize();

        if (depth == 0) {
            // Run on the first bounce
            generateGBuffer<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_intersections, dev_paths, dev_gBuffer);
        }

        depth++;

        shadeSimpleMaterials<<<numblocksPathSegmentTracing, blockSize1d>>> (
            iter, num_paths, dev_intersections, dev_paths, dev_materials);
        iterationComplete = depth == traceDepth;
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // CHECKITOUT: use dev_image as reference if you want to implement saving denoised images.
    // Otherwise, screenshots are also acceptable.
    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

void denoise(float c_phi, float n_phi, float p_phi) {
    std::cout << "Denoising with weights " << c_phi << " " << n_phi << " " << p_phi << std::endl;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMemcpy(dev_atrous_in, dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    checkCUDAError("Filling dev_atrous_in");

    const int blockSize1d = 128;
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    
    for (int i = 0; i < 1; i++) {
        float stepWidth = 1 << i;

        kernDenoise << <numBlocksPixels, blockSize1d >> > (
            cam.resolution.x, cam.resolution.y,
            dev_gBuffer, dev_image,
            dev_filter, dev_offsets,
            stepWidth, c_phi, n_phi, p_phi,
            dev_atrous_in, dev_atrous_out);

        glm::vec3* tmp = dev_atrous_in;
        dev_atrous_in = dev_atrous_out;
        dev_atrous_out = tmp;
    }
    cudaDeviceSynchronize();

    //cudaMemcpy(dev_image, dev_atrous_in, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
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

void showDenoise(uchar4* pbo) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, 10, dev_atrous_in);
}
