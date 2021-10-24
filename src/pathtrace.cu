#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/partition.h>


#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#include "../stream_compaction/efficient.cu"
#include "../denoise/oidn/apps/utils/image_io.h"

#include <iostream>

#include <OpenImageDenoise/oidn.hpp>

#include "options.h"

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

  if (x < resolution.x && y < resolution.y) {
    int index = x + (y * resolution.x);
    //float timeToIntersect = gBuffer[index].t * 256.0;

//        pbo[index].w = 0;
//        pbo[index].x = timeToIntersect;
//        pbo[index].y = timeToIntersect;
//        pbo[index].z = timeToIntersect;
    auto disp_attr = gBuffer[index].norm * 255.0f;


    pbo[index].w = 0;
    pbo[index].x = disp_attr.x;
    pbo[index].y = disp_attr.y;
    pbo[index].z = disp_attr.z;
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
// ...
static float* dev_kernel = NULL;
static glm::ivec2* dev_offsets = NULL;
static glm::vec3* dev_dn_sum = NULL;
static float* dev_dn_cum_w = NULL;

static oidn::DeviceRef oidn_device;
static oidn::FilterRef oidn_filter;
static int img_width, img_height;
static std::shared_ptr<oidn::ImageBuffer> denoised_img;

static glm::vec3 * dev_albedo_image = NULL;
static glm::vec3 * dev_normal_image = NULL;

static std::shared_ptr<oidn::ImageBuffer> albedo_image;
static std::shared_ptr<oidn::ImageBuffer> normal_image;

static Triangle * dev_tris = NULL;
static std::vector<int> geom_tris_starts;
static std::vector<int> geom_tris_ends;
static int * dev_tris_starts;
static int * dev_tris_ends;

static float dn_kernel[25] = {
        0.003765,	0.015019,	0.023792,	0.015019,	0.003765,
        0.015019,	0.059912,	0.094907,	0.059912,	0.015019,
        0.023792,	0.094907,	0.150342,	0.094907,	0.023792,
        0.015019,	0.059912,	0.094907,	0.059912,	0.015019,
        0.003765,	0.015019,	0.023792,	0.015019,	0.003765
};

static glm::ivec2 dn_offset[25] = {
        glm::ivec2(-2, -2), glm::ivec2(-2, -1), glm::ivec2(-2, 0), glm::ivec2(-2, 1), glm::ivec2(-2, 2),
        glm::ivec2(-1, -2), glm::ivec2(-1, -1), glm::ivec2(-1, 0), glm::ivec2(-1, 1), glm::ivec2(-1, 2),
        glm::ivec2(0, -2), glm::ivec2(0, -1), glm::ivec2(0, 0), glm::ivec2(0, 1), glm::ivec2(0, 2),
        glm::ivec2(1, -2), glm::ivec2(1, -1), glm::ivec2(1, 0), glm::ivec2(1, 1), glm::ivec2(1, 2),
        glm::ivec2(2, -2), glm::ivec2(2, -1), glm::ivec2(2, 0), glm::ivec2(2, 1), glm::ivec2(2, 2)
};


void pathtraceInit(Scene *scene) {
  hst_scene = scene;
  const Camera &cam = hst_scene->state.camera;
  const int pixelcount = cam.resolution.x * cam.resolution.y;

  cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
  cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  // pre calculate all of the offsets where the start of triangles are located, for each geom
  int tri_offset_tmp = 0;
  for(const auto &geom: scene->geoms){
    if (geom.type == MESH){
      geom_tris_starts.push_back(tri_offset_tmp);
      tri_offset_tmp += geom.num_tris;
      geom_tris_ends.push_back(tri_offset_tmp);
    }else{
      geom_tris_starts.push_back(-1);
      geom_tris_ends.push_back(-1);
    }
  }

  cudaMalloc(&dev_tris_starts, geom_tris_starts.size() * sizeof(int));
  cudaMemcpy(dev_tris_starts, geom_tris_starts.data(), geom_tris_starts.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(&dev_tris_ends, geom_tris_ends.size() * sizeof(int));
  cudaMemcpy(dev_tris_ends, geom_tris_ends.data(), geom_tris_ends.size() * sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc(&dev_tris, scene->triangles.size() * sizeof(Triangle));
  cudaMemcpy(dev_tris, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

  cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

  cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

  cudaMalloc(&dev_kernel, 25 * sizeof(float));
  cudaMemcpy(dev_kernel, &dn_kernel, 25 * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc(&dev_offsets, 25 * sizeof(glm::ivec2));
  cudaMemcpy(dev_offsets, &dn_offset, 25 * sizeof(glm::ivec2), cudaMemcpyHostToDevice);

  cudaMalloc(&dev_dn_sum, pixelcount * sizeof(glm::vec3));
  cudaMalloc(&dev_dn_cum_w, pixelcount * sizeof(float));

  // oidn
  img_width = cam.resolution.x;
  img_height = cam.resolution.y;
  oidn_device = oidn::newDevice();
  oidn_device.set("numThreads", OIDN_THREADS);
  oidn_device.commit();
  denoised_img = std::make_shared<oidn::ImageBuffer>(img_width, img_height, 3);
  albedo_image = std::make_shared<oidn::ImageBuffer>(img_width, img_height, 3);
  normal_image = std::make_shared<oidn::ImageBuffer>(img_width, img_height, 3);

  oidn_filter = oidn_device.newFilter("RT"); // generic ray tracing filter
  oidn_filter.setImage("color",  denoised_img->data(),  oidn::Format::Float3, img_width, img_height);
  oidn_filter.setImage("albedo", albedo_image->data(), oidn::Format::Float3, img_width, img_height);
  oidn_filter.setImage("normal", normal_image->data(), oidn::Format::Float3, img_width, img_height);
  oidn_filter.setImage("output", denoised_img->data(), oidn::Format::Float3, img_width, img_height);
  oidn_filter.set("hdr", true);
  oidn_filter.set("cleanAux", true);
  oidn_filter.commit();


  // TODO: initialize any extra device memeory you need

  cudaMalloc(&dev_albedo_image, pixelcount * sizeof(glm::vec3));
  cudaMalloc(&dev_normal_image, pixelcount * sizeof(glm::vec3));

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
  cudaFree(dev_albedo_image);
  cudaFree(dev_normal_image);

  cudaFree(dev_offsets);
  cudaFree(dev_kernel);
  cudaFree(dev_dn_sum);
  cudaFree(dev_dn_cum_w);

//    for(auto dev_ptr: dev_tris_l){
//      cudaFree(dev_ptr);
//    }
  cudaFree(dev_tris);

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


    // TODO: implement antialiasing by jittering the ray
    segment.ray.direction = glm::normalize(cam.view
                                           - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
                                           - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
    );

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, x, y);
    thrust::uniform_real_distribution<float> u01(0, 0.01 * ANTIALIAS_MULTIPLIER);
    auto ray_offset = calculateRandomDirectionInHemisphere2(glm::normalize(glm::cross(cam.up, cam.right)), rng);
    segment.ray.direction = glm::normalize(glm::vec3(segment.ray.direction.x + u01(rng), segment.ray.direction.y + u01(rng), segment.ray.direction.z + u01(rng)));

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
        , Triangle * tris
        , int * g_tris_starts
        , int * g_tris_ends
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
      }else if(geom.type == MESH){

#if CHECK_MESH_BOUNDING_BOXES
        // check bounding box first
        if (boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside) < 0.0f){
          continue;
        }
#endif

        // iterate through triangles of that mesh
        for(int j=g_tris_starts[i]; j<g_tris_ends[i]; j++){
          auto tri = tris[j];
          t =  triIntersectionTest(tri, pathSegment.ray, tmp_intersect, tmp_normal, outside);
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

        continue;
        //t = triIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
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
    }
    else
    {
      //The ray hits something
      intersections[path_index].t = t_min - EPSILON;
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
//__global__ void shadeFakeMaterial (
//  int iter
//  , int num_paths
//    , ShadeableIntersection * shadeableIntersections
//    , PathSegment * pathSegments
//    , Material * materials
//    )
//{
//  int idx = blockIdx.x * blockDim.x + threadIdx.x;
//  if (idx < num_paths)
//  {
//    ShadeableIntersection intersection = shadeableIntersections[idx];
//    if (intersection.t > 0.0f) { // if the intersection exists...
//      // Set up the RNG
//      // LOOK: this is how you use thrust's RNG! Please look at
//      // makeSeededRandomEngine as well.
//      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
//      thrust::uniform_real_distribution<float> u01(0, 1);
//
//      Material material = materials[intersection.materialId];
//      glm::vec3 materialColor = material.color;
//
//      // If the material indicates that the object was a light, "light" the ray
//      if (material.emittance > 0.0f) {
//        pathSegments[idx].color *= (materialColor * material.emittance);
//      }
//      // Otherwise, do some pseudo-lighting computation. This is actually more
//      // like what you would expect from shading in a rasterizer like OpenGL.
//      // TODO: replace this! you should be able to start with basically a one-liner
//      else {
//        float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
//        pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
//        pathSegments[idx].color *= u01(rng); // apply some noise because why not
//      }
//    // If there was no intersection, color the ray black.
//    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
//    // used for opacity, in which case they can indicate "no opacity".
//    // This can be useful for post-processing and image compositing.
//    } else {
//      pathSegments[idx].color = glm::vec3(0.0f);
//    }
//  }
//}

__global__ void shadeRealMaterial (
        int iter,
        int depth
        , int num_paths
        , ShadeableIntersection * shadeableIntersections
        , PathSegment * pathSegments
        , Material * materials
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {

    pathSegments[idx].remainingBounces -= 1;
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) { // if the intersection exists...
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.

      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
      //thrust::uniform_real_distribution<float> u01(-1.0, 1.0);


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

        //pathSegments[idx].color *= materialColor;

        glm::vec3 new_origin = pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t;

        scatterRay(
                pathSegments[idx],
                new_origin,
                intersection.surfaceNormal,
                material, rng);



/*
        How I did it before discovering that interactions.h existed.
        //glm::vec3 new_direction = glm::reflect(pathSegments[idx].ray.direction, intersection.surfaceNormal);
        glm::vec3 new_direction = intersection.surfaceNormal;
        new_direction = glm::clamp(glm::vec3(new_direction.x + u01(rng), new_direction.y + u01(rng), new_direction.z + u01(rng)), -1.0f, 1.0f);
        //new_direction = new_direction + u01(rng);

        pathSegments[idx].ray.origin = new_origin;
        pathSegments[idx].ray.direction = new_direction;

 */

//        float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
//        pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
//        pathSegments[idx].color *= u01(rng); // apply some noise because why not
      }
      // If there was no intersection, color the ray black.
      // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
      // used for opacity, in which case they can indicate "no opacity".
      // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].color = glm::vec3(0.0f, 0.0f, 0.0f);
      pathSegments[idx].remainingBounces = 0;
    }
  }
}

__global__ void updateGBuffer (
        int num_px,
        glm::vec3* image,
        GBufferPixel* gBuffer,
        glm::vec3* sum,
        float* cum_w
) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_px)
  {
    gBuffer[idx].color = image[idx];
    sum[idx].r = 0.0f;
    sum[idx].g = 0.0f;
    sum[idx].b = 0.0f;
    cum_w[idx] = 0.0f;
  }
}

__global__ void generateGBuffer (
        int num_paths,
        ShadeableIntersection* shadeableIntersections,
        PathSegment* pathSegments,
        Material * materials,
        GBufferPixel* gBuffer) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    gBuffer[idx].t = shadeableIntersections[idx].t;

    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment iterationPath = pathSegments[idx];

    gBuffer[idx].norm = glm::abs(intersection.surfaceNormal);
    gBuffer[idx].pos = glm::abs(glm::normalize(pathSegments[idx].ray.origin + (glm::normalize(pathSegments[idx].ray.direction) * intersection.t)));

  }


}

__device__ int _2d21d(const glm::vec2 &xy, int width){
  return  floorf(xy.x) + floorf(width* xy.y);
}

__device__ glm::vec2 _1d22d(int i, int width){
  return glm::vec2(floorf(i % width), floorf(i / width));
}

__global__ void kern_actuallyDenoise (
        int num_px,
        int offset_idx,
        glm::vec3* sum,
        float* cum_w,
        float resX,
        int resXi,
        int resYi,
        float resY,
        float c_phi,
        float n_phi,
        float p_phi,
        int stepwidth,
        glm::ivec2* offset,
        float* kernel,
        GBufferPixel * gbuffer,
        glm::vec3 * image
){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //if (idx == 70000) //(idx < num_px)
  if (idx < num_px)
  {
//    image[idx].r = 255.0;
//    image[idx].g = 1.0;
//    image[idx].b = 1.0;

    // get x/y PX from IDX
    //auto pxy = _1d22d(idx, resX);
    //int resXi = floor(resX);
    //int resYi = floor(resY);

    //glm::ivec2 pxy = glm::ivec2(idx % resXi, idx / resXi);
    int px = idx % resXi;
    int py = idx / resXi;



    //glm::vec2 step = glm::vec2(1.0f/resX, 1.0f/resY );

    glm::vec3 cval = gbuffer[idx].color;
    glm::vec3 nval = gbuffer[idx].norm;
    glm::vec3 pval = gbuffer[idx].pos;

    //printf("==============================================================================================\n");

//    printf("pxy ; %i %i\n", pxy.x, pxy.y);
//    printf("step ; %g %g\n", step.x, step.y);
//    printf("cval ; %g %g %g\n", cval.x, cval.y, cval.z);
//    printf("nval ; %g %g %g\n", nval.x, nval.y, nval.z);
//    printf("pval ; %g %g %g\n", pval.x, pval.y, pval.z);

    //glm::vec3 sum = glm::vec3(0, 0, 0);
    //float cum_w = 0.0;

    //for(int i = 0; i < 25; i++) {
    auto step = stepwidth/5;
    //printf("%i-------------------------------------------------------\n", i);
    int u = px + offset[offset_idx].x*(step);
    int v = py + offset[offset_idx].y*(step);

    u = glm::clamp(u, 0, resXi-1);
    v = glm::clamp(v, 0, resYi-1);
    //if (!(u < 0 or u > resXi or v < 0 or v > resYi)){


    // check what goes on between 20 and 21 here.

//      uv.x = glm::clamp(uv.x, 0.0f, (float) resX);
//      uv.y = glm::clamp(uv.y, 0.0f, (float) resY);
    int uvxyi = u + (resXi* v);




//      uv.x = glm::clamp(uv.x, 0.0f, (float) resX);
//      uv.y = glm::clamp(uv.y, 0.0f, (float) resY);
    //int uvxyi = floor(uv.x) + floor(resX* uv.y);

    glm::vec3 ctmp = gbuffer[uvxyi].color;
#if DENOISER_WEIGHTING_ENABLE
    glm::vec3 t = cval - ctmp;
    float dist2 = glm::dot(t,t);
    float c_w = glm::min(glm::exp(-(dist2)/(c_phi*c_phi)), 1.0f);

    //printf("ctmp ; %g %g %g  ; t %g %g %g ; dist2 %g ; c_w %g \n", ctmp.x, ctmp.y, ctmp.z, t.x, t.y, t.z, dist2, c_w);


    glm::vec3 ntmp = gbuffer[uvxyi].norm;
    t = nval - ntmp;
    dist2 = fmaxf(glm::dot(t,t)/(step*step),0.0f);
    float n_w = glm::min(glm::exp(-(dist2)/(n_phi*n_phi)), 1.0f);

    //printf("ntmp ; %g %g %g  ; t %g %g %g ; dist2 %g ; n_w %g \n", ntmp.x, ntmp.y, ntmp.z, t.x, t.y, t.z, dist2, n_w);

    glm::vec3 ptmp = gbuffer[uvxyi].pos;
    t = pval - ptmp;
    dist2 = glm::dot(t,t);
    float p_w = glm::min(glm::exp(-(dist2)/(p_phi*p_phi)),1.0f);

    //printf("ptmp ; %g %g %g  ; t %g %g %g ; dist2 %g ; p_w %g \n", ptmp.x, ptmp.y, ptmp.z, t.x, t.y, t.z, dist2, p_w);

    float weight = c_w * n_w * p_w;
    sum[idx] += ctmp * weight* kernel[offset_idx];
    cum_w[idx] += weight * kernel[offset_idx];
#else
    sum[idx] += ctmp * kernel[offset_idx];
        cum_w[idx] +=  kernel[offset_idx];
#endif
    //weight = 0;


//      if(weight <= 0){
//        sum[idx] += ctmp * kernel[offset_idx];
//        cum_w[idx] += kernel[offset_idx];
//      }else{
//        sum[idx] += ctmp * weight * kernel[offset_idx];
//        cum_w[idx] += weight*kernel[offset_idx];
//      }




    //printf("uvxy ; %i %i  ; uvxyi %i\n", u, v, uvxyi);








    //printf("-------------------------------------------------------\n");

    //}
//
    //printf("==============================================================================================\n");
    if(offset_idx == 24){


      image[idx] = sum[idx]/cum_w[idx];

      //printf("FINAL %g %g %g ; %g ; %g %g %g \n", sum[idx].r,sum[idx].g,sum[idx].b , cum_w[idx], image[idx].r, image[idx].g, image[idx].b);
    }

    //image[idx] = cval;

  }
  /*
   * uniform sampler2D colorMap, normalMap, posMap;
    uniform float c_phi, n_phi, p_phi, stepwidth;
    uniform float kernel[25];
    uniform vec2 offset[25];
    void main(void) {
      vec4 sum = vec4(0.0);
      vec2 step = vec2(1./512., 1./512.); // resolution
      vec4 cval = texture2D(colorMap, gl_TexCoord[0].st);
      vec4 nval = texture2D(normalMap, gl_TexCoord[0].st);
      vec4 pval = texture2D(posMap, gl_TexCoord[0].st);
      float cum_w = 0.0;
      for(int i = 0; i < 25; i++) {
        vec2 uv = gl_TexCoord[0].st + offset[i]*step*stepwidth;
        vec4 ctmp = texture2D(colorMap, uv);
        vec4 t = cval - ctmp;

        float dist2 = dot(t,t);
        float c_w = min(exp(-(dist2)/c_phi), 1.0);
        vec4 ntmp = texture2D(normalMap, uv);
        t = nval - ntmp;
        dist2 = max(dot(t,t)/(stepwidth*stepwidth),0.0);
        float n_w = min(exp(-(dist2)/n_phi), 1.0);
        vec4 ptmp = texture2D(posMap, uv);
        t = pval - ptmp;
        dist2 = dot(t,t);
        float p_w = min(exp(-(dist2)/p_phi),1.0);
        float weight = c_w * n_w * p_w;
        sum += ctmp * weight * kernel[i];
        cum_w += weight*kernel[i];
      }
      gl_FragData[0] = sum/cum_w;
    }
   */



}

__global__ void kern_actuallyDenoiseNoWeight (
        int num_px,
        int offset_idx,
        glm::vec3* sum,
        float* cum_w,
        float resX,
        int resXi,
        int resYi,
        float resY,
        float stepwidth,
        glm::vec2* offset,
        float* kernel,
        GBufferPixel * gbuffer,
        glm::vec3 * image
){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //if (idx == 70000) //(idx < num_px)
  if (idx < num_px)
  {


    glm::ivec2 pxy = glm::ivec2(idx % resXi, idx / resXi);


    glm::vec2 step = glm::vec2(1.0f/resX, 1.0f/resY );
    //glm::vec3 cval = gbuffer[idx].color;

    //printf("==============================================================================================\n");

//    printf("%i-------------------------------------------------------\n", offset_idx);
//    printf("pxy ; %i %i\n", pxy.x, pxy.y);
//    printf("step ; %g %g\n", step.x, step.y);

    //printf("cval ; %g %g %g\n", cval.x, cval.y, cval.z);
    //printf("nval ; %g %g %g\n", nval.x, nval.y, nval.z);
    //printf("pval ; %g %g %g\n", pval.x, pval.y, pval.z);

    //glm::vec3 sum = glm::vec3(0, 0, 0);
    //float cum_w = 0.0;

    //for(int i = 0; i < 25; i++) {

    /*
     * 4 ; 4*200/800 = 1
     * 8 ; 1/4 * 8 = 2
     * 12 ; 184 * 12 = 3
     */

    int u = pxy.x + floor(offset[offset_idx].x*((step.x*stepwidth*32) + 1.0f));
    int v = pxy.y + floor(offset[offset_idx].y*((step.y*stepwidth*32) + 1.0f));


    //glm::ivec2 uv = pxy + (offset[offset_idx]*step*stepwidth);
    if (!(u < 0 or u > resXi or v < 0 or v > resYi)){


      // check what goes on between 20 and 21 here.

//      uv.x = glm::clamp(uv.x, 0.0f, (float) resX);
//      uv.y = glm::clamp(uv.y, 0.0f, (float) resY);
      int uvxyi = u + (resXi* v);

      //printf("uvxy ; %i %i  ; uvxyi %i\n", u, v, uvxyi);

      glm::vec3 ctmp = gbuffer[uvxyi].color;

      sum[idx] += ctmp * kernel[offset_idx];
      cum_w[idx] += kernel[offset_idx];
    }



    //printf("-------------------------------------------------------\n");

    //}
//
    //printf("==============================================================================================\n");
    if(offset_idx == 24){
      image[idx] = sum[idx]/cum_w[idx];
    }

    //image[idx] = cval;

  }


}

__global__ void kern_saveAuxDenoiseData (
        int num_paths
        , ShadeableIntersection * shadeableIntersections,
        PathSegment * iterationPaths
        , Material * materials,
        glm::vec3 * albedo,
        glm::vec3 * normal
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths) {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment iterationPath = iterationPaths[idx];

    if (intersection.t > 0.0f) {
      Material material = materials[intersection.materialId];
      albedo[iterationPath.pixelIndex] = material.color;
      normal[iterationPath.pixelIndex] = intersection.surfaceNormal;
    }else{
      albedo[iterationPath.pixelIndex] = glm::vec3(0.0f, 0.0f, 0.0f);
      normal[iterationPath.pixelIndex] = glm::vec3(0.0f, 0.0f, 0.0f);
    }
  }
}



struct is_not_zero_remaining_bounces
{
    __host__ __device__
    bool operator()(const PathSegment& path)
    {
      return (path.remainingBounces > 0);
    }
};

struct sort_path_by_mat
{
    __host__ __device__
    bool operator()(const PathSegment& path)
    {
      return (path.remainingBounces > 0);
    }
};

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
void pathtrace(uchar4 *pbo, int frame, int iter) {
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

  const dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;

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
  int original_num_paths = num_paths;

  thrust::device_ptr<PathSegment> dev_thrust_paths(dev_paths);

  thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections(dev_intersections);

  //thrust::device_ptr<PathSegment> dev_thrust_paths_read(dev_paths_read);
  //thrust::device_ptr<PathSegment> dev_thrust_paths_write(dev_paths_write);
  //thrust::copy(dev_thrust_paths, dev_thrust_paths+num_paths, dev_thrust_paths_read);

  // --- PathSegment Tracing Stage ---
  // Shoot ray into scene, bounce between objects, push shading chunks

  // Empty gbuffer
  cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));

  // clean shading chunks
  cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

  bool iterationComplete = false;
  while (!iterationComplete) {

    //  tracing
    dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
    computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth
            , num_paths
            , dev_paths
            , dev_geoms
            , hst_scene->geoms.size()
            , dev_tris
            , dev_tris_starts
            , dev_tris_ends
            , dev_intersections
    );
    checkCUDAError("trace one bounce");
    cudaDeviceSynchronize();

    if (depth == 0) {
      generateGBuffer<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_intersections, dev_paths, dev_materials, dev_gBuffer);
    }



    // TODO:
    // --- Shading Stage ---
    // Shade path segments based on intersections and generate new rays by
    // evaluating the BSDF.
    // Start off with just a big kernel that handles all the different
    // materials you have in the scenefile.
    // TODO: compare between directly shading the path segments and shading
    // path segments that have been reshuffled to be contiguous in memory.

    shadeRealMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
            frame, depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
    );



    // populate denoise images
#if ENABLE_OIDN
    if(depth == 0){
    kern_saveAuxDenoiseData<<<numblocksPathSegmentTracing, blockSize1d>>> (
                    num_paths,
                    dev_intersections,
                            dev_paths,
                    dev_materials,
                    dev_albedo_image,
                    dev_normal_image
    );
  }
#endif

//std::cout << "num paths at iter " << depth << " is " << num_paths  << '\n';
    //std::cout << num_paths  << '\n';

    // use thrust to remove rays with zero bounces remaining
    auto compact_end = thrust::partition(dev_thrust_paths, dev_thrust_paths+num_paths,
                                         is_not_zero_remaining_bounces());

    num_paths = compact_end - dev_thrust_paths;
//




#if ENABLE_MATERIAL_SORTING
    // use thrust to in place sort the remaining paths with respect to material ID
  thrust::sort_by_key(dev_thrust_intersections, dev_thrust_intersections+num_paths, dev_thrust_paths);
#endif


//
//  auto flip1 = dev_paths_write;
//  auto flip2 = dev_thrust_paths_write;
//  dev_paths_write = dev_paths;
//  dev_thrust_paths_write = dev_thrust_paths;
//  dev_paths = flip1;
//  dev_thrust_paths = flip2;

    if(num_paths == 0){
      iterationComplete = true;
    }

    depth++;
    // TODO: should be based off stream compaction results.

    // guard
//  if(depth > 10){
//    std::cout << "GUARD CALLED " << '\n';
//    iterationComplete = true;
//  }
  }

  //std::cout << "ITER \n";


  // Assemble this iteration and apply it to the image

  finalGather<<<numBlocksPixels, blockSize1d>>>(original_num_paths, dev_image, dev_paths);

  ///////////////////////////////////////////////////////////////////////////

  // CHECKITOUT: use dev_image as reference if you want to implement saving denoised images.
  // Otherwise, screenshots are also acceptable.
  // Retrieve image from GPU
//    cudaMemcpy(hst_scene->state.image.data(), dev_image,
//            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

#if ENABLE_OIDN
  // Retrieve image from GPU (pre denoise)
    cudaMemcpy(denoised_img->data(), dev_image,
               pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    // retrieve aux images from GPU
    cudaMemcpy(albedo_image->data(), dev_albedo_image,
               pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(normal_image->data(), dev_normal_image,
               pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);


    // Filter the image
    oidn_filter.execute();

    // Check for errors
    const char* errorMessage;
    if (oidn_device.getError(errorMessage) != oidn::Error::None){
      std::cout << "Error: " << errorMessage << std::endl;
    }else{
      // put denoised image back on gpu
      cudaMemcpy(dev_image, denoised_img->data(),
                 pixelcount * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    }
#endif

  // call denoise
  //for(int iter_idx=0; iter_idx<25; iter_idx++){

//    std::cout << hst_scene->dn_colorWeight << '\n';
//    std::cout << hst_scene->dn_normalWeight << '\n';
//    std::cout << hst_scene->dn_positionWeight << '\n';

  if(hst_scene->denoise){

    updateGBuffer<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_gBuffer, dev_dn_sum, dev_dn_cum_w);
    //cudaMemcpy(dev_offsets, &dn_offset, 25 * sizeof(glm::vec2), cudaMemcpyHostToDevice);
    // kern_actuallyDenoise<<<numBlocksPixels, blockSize1d>>>(
    for(int iter_idx=0; iter_idx<25; iter_idx++) {
      kern_actuallyDenoise<<<numBlocksPixels, blockSize1d>>>(
              pixelcount,
              iter_idx,
              dev_dn_sum,
              dev_dn_cum_w,
              hst_scene->state.camera.resolution.x,
              glm::floor(hst_scene->state.camera.resolution.x),
              glm::floor(hst_scene->state.camera.resolution.y),
              hst_scene->state.camera.resolution.y,
              hst_scene->dn_colorWeight,
              hst_scene->dn_normalWeight,
              hst_scene->dn_positionWeight,
              floor(hst_scene->dn_filterSize),
              dev_offsets,
              dev_kernel,
              dev_gBuffer,
              dev_image);

//        kern_actuallyDenoiseNoWeight<<<numBlocksPixels, blockSize1d>>>(
//                pixelcount,
//                iter_idx,
//                dev_dn_sum,
//                dev_dn_cum_w,
//                hst_scene->state.camera.resolution.x,
//                glm::floor(hst_scene->state.camera.resolution.x),
//                glm::floor(hst_scene->state.camera.resolution.y),
//                hst_scene->state.camera.resolution.y,
//                hst_scene->dn_filterSize,
//                dev_offsets,
//                dev_kernel,
//                dev_gBuffer,
//                dev_image);
    }
  }





  // Retrieve image from GPU
  // if(!hst_scene->disp_idx == 0){
  // Send results to OpenGL buffer for rendering
  sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
  cudaMemcpy(hst_scene->state.image.data(), dev_image,
             pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
//  }else{
//    gbufferToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, dev_gBuffer);
//    cudaMemcpy(hst_scene->state.image.data(), dev_gBuffer,
//               pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
//  }


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

void showImage(uchar4* pbo, int iter) {
  const Camera &cam = hst_scene->state.camera;
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d(
          (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
          (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

  // Send results to OpenGL buffer for rendering
  sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
}
