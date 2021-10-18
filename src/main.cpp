#include "main.h"
#include "preview.h"
#include <cstring>
#include "profile_log/logCore.hpp"
#include "denoise.h"
#include "../imgui/imgui.h"
#include "../imgui/imgui_impl_glfw.h"
#include "../imgui/imgui_impl_opengl3.h"


#pragma warning(push)
#pragma warning(disable:4996)

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

// CHECKITOUT: simple UI parameters.
// Search for any of these across the whole project to see how these are used,
// or look at the diff for commit 1178307347e32da064dce1ef4c217ce0ca6153a8.
// For all the gory GUI details, look at commit 5feb60366e03687bfc245579523402221950c9c5.
int ui_iterations = 0;
int startupIterations = 0;
int lastLoopIterations = 0;
bool ui_showGbuffer = false;
bool ui_denoise = false;
bool ui_temporal = false;
int ui_filterSize = 8;//80;
float ui_colorWeight = 0.6f;//0.45f;
float ui_normalWeight = 0.9f;//0.35f;
float ui_positionWeight = 10.f;//0.2f;
int ui_denoiseTypeIndex = static_cast<int>(Denoise::DenoiserType::A_TROUS);
int ui_gBufferDataIndex = static_cast<int>(GBufferDataType::TIME);
bool ui_saveAndExit = false;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene *scene;
RenderState *renderState;
int iteration;
bool paused;

int width;
int height;

#if ENABLE_CACHE_FIRST_INTERSECTION
extern bool cacheFirstIntersection;
extern bool firstIntersectionCached;
#endif // ENABLE_CACHE_FIRST_INTERSECTION
#if ENABLE_PROFILE_LOG
bool saveProfileLog = false;
#endif // ENABLE_PROFILE_LOG

//-------------------------------
//-------------MAIN--------------
//-------------------------------

extern void unitTest();

int main(int argc, char** argv) {
    startTimeString = currentTimeString();

    std::string sceneFileStr;
    const char* sceneFile = nullptr;

    if (argc < 2) {
        //printf("Usage: %s SCENEFILE.txt\n", argv[0]);
        //return 1;
        //sceneFile = "../scenes/cornell_testOutline.txt";
        //sceneFile = "../scenes/cornell.txt";
        //sceneFile = "../scenes/cornellMF.txt";
        //sceneFile = "../scenes/cornell2.txt";
        //sceneFile = "../scenes/sphere.txt";
        //sceneFile = "../scenes/cornell_ramp.txt";

        //sceneFile = "../scenes/PA_BVH2000.txt";
        //sceneFile = "../scenes/PA_BVH135280.txt";

        //sceneFile = "../scenes/cornell_garage_kit.txt";
        //sceneFile = "../scenes/cornell_garage_kit_microfacet.txt";
        std::cout << "Input sceneFile: " << std::flush;
        std::cin >> sceneFileStr;
        sceneFile = sceneFileStr.c_str();
    }
    else {
        sceneFile = argv[1];
    }
    if (argc < 3) {
        std::cout << "Enable denoise? (1/0): " << std::flush;
        std::cin >> ui_denoiseTypeIndex;
        ui_denoise = ui_denoiseTypeIndex;
        ui_denoiseTypeIndex = glm::clamp(ui_denoiseTypeIndex, 1, static_cast<int>(Denoise::DenoiserType::MAX_INDEX)) - 1;
    }
    else {
        ui_denoiseTypeIndex = atoi(argv[2]);
        ui_denoise = ui_denoiseTypeIndex;
        ui_denoiseTypeIndex = glm::clamp(ui_denoiseTypeIndex, 1, static_cast<int>(Denoise::DenoiserType::MAX_INDEX)) - 1;
    }

    if (argc < 4) {
        std::cout << "Filter size? : " << std::flush;
        std::cin >> ui_filterSize;
    }
    else {
        ui_filterSize = atoi(argv[3]);
    }

#if ENABLE_PROFILE_LOG
    if (argc < 5) {
        std::cout << "Save profile log? (1/0): " << std::flush;
        std::cin >> saveProfileLog;
    }
    else {
        saveProfileLog = atoi(argv[4]);
    }
#endif // ENABLE_PROFILE_LOG

    // Load scene file
    scene = new Scene(sceneFile);

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    ui_iterations = renderState->iterations;
    startupIterations = ui_iterations;

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    cameraPosition = cam.position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
    glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
    phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
    theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

    // Initialize CUDA and GL components
    init();

    unitTest();

    // GLFW main loop
    mainLoop();

    delete scene;
    pathtraceFree();
    cudaDeviceSynchronize();
    return 0;
}

void saveImage() {
    float samples = static_cast<float>(iteration);
    // output image file
    Image::image img(width, height);

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
#if false//!PREGATHER_FINAL_IMAGE
            img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
#else // PREGATHER_FINAL_IMAGE
            img.setPixel(width - 1 - x, y, glm::vec3(pix));
#endif // PREGATHER_FINAL_IMAGE
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

bool triggerClearIterationByUI() {
    static bool last_ui_denoise = ui_denoise;
    static bool last_ui_temporal = ui_temporal;
    static int last_ui_filterSize = ui_filterSize;
    static int last_ui_denoiseTypeIndex = ui_denoiseTypeIndex;

    static float last_ui_colorWeight = ui_colorWeight;
    static float last_ui_normalWeight = ui_normalWeight;
    static float last_ui_positionWeight = ui_positionWeight;

    bool result = false;

    if (ui_denoise != last_ui_denoise) {
        last_ui_denoise = ui_denoise;
        result = true;
    }
    if (ui_temporal != last_ui_temporal) {
        last_ui_temporal = ui_temporal;
        if (ui_denoise) {
            result = true;
        }
    }
    if (ui_filterSize != last_ui_filterSize) {
        last_ui_filterSize = ui_filterSize;
        if (ui_denoise) {
            result = true;
        }
    }
    if (ui_denoiseTypeIndex != last_ui_denoiseTypeIndex) {
        last_ui_denoiseTypeIndex = ui_denoiseTypeIndex;
        if (ui_denoise) {
            result = true;
        }
    }
    if (ui_colorWeight != last_ui_colorWeight) {
        last_ui_colorWeight = ui_colorWeight;
        if (ui_denoise) {
            result = true;
        }
    }
    if (ui_normalWeight != last_ui_normalWeight) {
        last_ui_normalWeight = ui_normalWeight;
        if (ui_denoise) {
            result = true;
        }
    }
    if (ui_positionWeight != last_ui_positionWeight) {
        last_ui_positionWeight = ui_positionWeight;
        if (ui_denoise) {
            result = true;
        }
    }
    return result;
}

void runCuda() {
    if (triggerClearIterationByUI()) {
        iteration = 0;
    }

    if (lastLoopIterations != ui_iterations) {
      lastLoopIterations = ui_iterations;
      camchanged = true;
    }

    if (camchanged) {
        iteration = 0;
        Camera &cam = renderState->camera;
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        cam.view = -glm::normalize(cameraPosition);
        glm::vec3 v = cam.view;
        glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
        glm::vec3 r = glm::cross(v, u);
        cam.up = glm::cross(r, v);
        cam.right = r;

        cam.position = cameraPosition;
        cameraPosition += cam.lookAt;
        cam.position = cameraPosition;
        camchanged = false;
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (iteration == 0) {
        pathtraceFree();
        pathtraceInit(scene);
    }

    uchar4 *pbo_dptr = NULL;
    cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

    if (iteration < ui_iterations) {
        iteration++;

        // execute the kernel
        int frame = 0;
        pathtrace(pbo_dptr, frame, iteration);
    }

    if (ui_showGbuffer) {
      showGBuffer(pbo_dptr, static_cast<GBufferDataType>(ui_gBufferDataIndex));
    } else {
      showImage(pbo_dptr, iteration);

      //cudaMemcpy(scene->state.image.data(), scene->dev_frameBuffer.buffer,
      //    renderState->camera.resolution.x * renderState->camera.resolution.y * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    }

    // unmap buffer object
    cudaGLUnmapBufferObject(pbo);

    if (ui_saveAndExit) {
        saveImage();
        pathtraceFree();
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        switch (key) {
        case GLFW_KEY_P:
            paused = !paused;
            break;
        case GLFW_KEY_ESCAPE:
            saveImage();
            glfwSetWindowShouldClose(window, GL_TRUE);
            break;
        case GLFW_KEY_S:
            saveImage();
            break;
        case GLFW_KEY_SPACE:
            camchanged = true;
            renderState = &scene->state;
            renderState->camera.lookAt = ogLookAt;
            break;
        case GLFW_KEY_UP:
            camchanged = true;
            //renderState = &scene->state;
            //renderState->camera.lookAt = ogLookAt;
            ++renderState->traceDepth;
            break;
        case GLFW_KEY_DOWN:
            camchanged = true;
            //renderState = &scene->state;
            //renderState->camera.lookAt = ogLookAt;
            renderState->traceDepth = std::max(0, renderState->traceDepth - 1);
            break;
        case GLFW_KEY_RIGHT:
            camchanged = true;
            //renderState = &scene->state;
            //renderState->camera.lookAt = ogLookAt;
            renderState->recordDepth = std::min(renderState->traceDepth, renderState->recordDepth + 1);
            break;
        case GLFW_KEY_LEFT:
            camchanged = true;
            //renderState = &scene->state;
            //renderState->camera.lookAt = ogLookAt;
            renderState->recordDepth = std::max(-1, renderState->recordDepth - 1);
            break;
#if ENABLE_CACHE_FIRST_INTERSECTION
        case GLFW_KEY_C:
            cacheFirstIntersection = !cacheFirstIntersection;
            firstIntersectionCached = false;
            break;
#endif // ENABLE_CACHE_FIRST_INTERSECTION
        case GLFW_KEY_0:
        case GLFW_KEY_1:
        case GLFW_KEY_2:
        case GLFW_KEY_3:
        case GLFW_KEY_4:
        case GLFW_KEY_5:
        case GLFW_KEY_6:
        case GLFW_KEY_7:
        case GLFW_KEY_8:
        case GLFW_KEY_9:
        {
            size_t index = key - GLFW_KEY_0;
            if (index < scene->postprocesses.size()) {
                scene->postprocesses[index].second = !scene->postprocesses[index].second;
            }
        }
            break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  if (ImGui::GetIO().WantCaptureMouse) return;
  leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
  rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
  if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
  if (leftMousePressed) {
    // compute new camera parameters
    phi -= static_cast<float>(xpos - lastX) / width;
    theta -= static_cast<float>(ypos - lastY) / height;
    theta = std::fmax(0.001f, std::fmin(theta, PI));
    camchanged = true;
  }
  else if (rightMousePressed) {
    zoom += static_cast<float>(ypos - lastY) / height;
    zoom = std::fmax(0.1f, zoom);
    camchanged = true;
  }
  else if (middleMousePressed) {
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    glm::vec3 forward = cam.view;
    forward.y = 0.0f;
    forward = glm::normalize(forward);
    glm::vec3 right = cam.right;
    right.y = 0.0f;
    right = glm::normalize(right);

    cam.lookAt -= (float) (xpos - lastX) * right * 0.01f;
    cam.lookAt += (float) (ypos - lastY) * forward * 0.01f;
    camchanged = true;
  }
  lastX = xpos;
  lastY = ypos;
}

void unitTest() {
    printf("---Start Unit Test---\n");
    glm::vec3 a(0.f, 1.f, 0.f);
    glm::vec3 b(1.f, -1.f, 2.f);
    glm::vec3 c = glm::max(a, b);
    printf("c = glm::max(<%f,%f,%f>,<%f,%f,%f>) = <%f,%f,%f>\n",
        a.r, a.g, a.b,
        b.r, b.g, b.b,
        c.r, c.g, c.b);
    printf("---End Unit Test---\n");
}

#pragma warning(pop)
