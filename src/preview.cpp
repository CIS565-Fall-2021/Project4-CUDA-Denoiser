#define _CRT_SECURE_NO_DEPRECATE
#include <ctime>
#include <iomanip>
#include "main.h"
#include "preview.h"
#include "denoise.h"
#include "profile_log/logCore.hpp"
#include "../imgui/imgui.h"
#include "../imgui/imgui_impl_glfw.h"
#include "../imgui/imgui_impl_opengl3.h"


#pragma warning(push)
#pragma warning(disable:4996)

#define IMGUI_IMPL_OPENGL_LOADER_GLEW

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

GLFWwindow *window;

#if ENABLE_CACHE_FIRST_INTERSECTION
extern bool cacheFirstIntersection;
#endif // ENABLE_CACHE_FIRST_INTERSECTION

std::string currentTimeString() {
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];

    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", localtime(&now));
    //strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));

    return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

static const std::unordered_map<Denoise::DenoiserType, std::string> denoiserTypeToString {
    { Denoise::DenoiserType::A_TROUS, "A Trous" },
    { Denoise::DenoiserType::A_TROUS_EDGE_AVOIDING, "Edge Avoiding" },
    { Denoise::DenoiserType::A_TROUS_EDGE_AVOIDING_MORE_PARAM, "More Param" },
};

static const std::unordered_map<GBufferDataType, std::string> gBufferDataTypeToString {
    { GBufferDataType::TIME, "Time" },
    { GBufferDataType::BASE_COLOR, "Base Color" },
    { GBufferDataType::NORMAL, "Normal" },
    { GBufferDataType::OBJECT_ID, "Object ID" },
    { GBufferDataType::MATERIAL_ID, "Material ID" },
    { GBufferDataType::POSITION, "Position" },
};

static ImGuiWindowFlags windowFlags= ImGuiWindowFlags_None | ImGuiWindowFlags_NoMove;
static bool ui_hide = false;

void initTextures() {
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void) {
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    GLfloat texcoords[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader() {
    const char *attribLocations[] = { "Position", "Texcoords" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    //glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1) {
        glUniform1i(location, 0);
    }

    return program;
}

void deletePBO(GLuint* pbo) {
    if (pbo) {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint* tex) {
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void cleanupCuda() {
    if (pbo) {
        deletePBO(&pbo);
    }
    if (displayImage) {
        deleteTexture(&displayImage);
    }
}

void initCuda() {
    cudaGLSetGLDevice(0);

    // Clean up on program exit
    atexit(cleanupCuda);
}

void initPBO() {
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);

}

void errorCallback(int error, const char* description) {
    fprintf(stderr, "%s\n", description);
}

bool init() {
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }

    // Initialize other stuff
    initVAO();
    initTextures();
    initCuda();
    initPBO();
    scene->execInitCallbacks();
    checkCUDAError("initOtherStuff");
    GLuint passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    //// Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    return true;
}

bool denoiseItemsGetter(void* data, int idx, const char** outText) {
    auto& dataTypes = *static_cast<const std::unordered_map<Denoise::DenoiserType, std::string>*>(data);
    if (dataTypes.empty()) {
        return false;
    }

    auto key = static_cast<Denoise::DenoiserType>(idx);
    auto iter = dataTypes.find(key);
    if (iter == dataTypes.end()) {
        *outText = dataTypes.begin()->second.c_str();
    }
    else {
        *outText = iter->second.c_str();
    }
    return true;
}

bool gBufferItemsGetter(void* data, int idx, const char** outText) {
    auto& dataTypes = *static_cast<const std::unordered_map<GBufferDataType, std::string>*>(data);
    if (dataTypes.empty()) {
        return false;
    }

    auto key = static_cast<GBufferDataType>(idx);
    auto iter = dataTypes.find(key);
    if (iter == dataTypes.end()) {
        *outText = dataTypes.begin()->second.c_str();
    }
    else {
        *outText = iter->second.c_str();
    }
    return true;
}

void drawGui(int windowWidth, int windowHeight) {
    // Dear imgui new frame
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Dear imgui define
    ImVec2 minSize(300.f, 220.f); // (300.f, 300.f);
    ImVec2 maxSize((float)windowWidth * 0.5, (float)windowHeight * 0.3);
    ImGui::SetNextWindowSizeConstraints(minSize, maxSize);

    ImGui::SetNextWindowPos(ui_hide ? ImVec2(-1000.f, -1000.f) : ImVec2(0.0f, 0.0f));

    ImGui::Begin("Control Panel", 0, windowFlags);
    ImGui::SetWindowFontScale(1);

    ImGui::Text("press H to hide GUI completely.");
    if (ImGui::IsKeyPressed('H')) {
        ui_hide = !ui_hide;
    }

    ImGui::SliderInt("Iterations", &ui_iterations, 1, startupIterations);

    ImGui::Checkbox("Denoise", &ui_denoise);
    ImGui::Checkbox("Temporal", &ui_temporal);

    ImGui::Combo("Denoise Type", &ui_denoiseTypeIndex, denoiseItemsGetter, const_cast<void*>(reinterpret_cast<const void*>(&denoiserTypeToString)), static_cast<int>(denoiserTypeToString.size()));

    ImGui::SliderInt("Filter Size", &ui_filterSize, 0, 100);
    ImGui::SliderFloat("Color Weight", &ui_colorWeight, 0.0f, 20.0f);
    ImGui::SliderFloat("Normal Weight", &ui_normalWeight, 0.0f, 20.0f);
    ImGui::SliderFloat("Position Weight", &ui_positionWeight, 0.0f, 50.0f);

    ImGui::Separator();

    ImGui::Checkbox("Show GBuffer", &ui_showGbuffer);

    //ImGui::BeginCombo("GBuffer Data", "Preview?");
    //ImGui::Combo("GBuffer Data", &ui_gBufferDataIndex, { "Time" });
    //ImGui::EndCombo();

    ImGui::Combo("GBuffer Data Type", &ui_gBufferDataIndex, gBufferItemsGetter, const_cast<void*>(reinterpret_cast<const void*>(&gBufferDataTypeToString)), static_cast<int>(gBufferDataTypeToString.size()));

    ImGui::Separator();

    if (ImGui::Button("Save image and exit")) {
        ui_saveAndExit = true;
    }

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

extern bool paused;
#if ENABLE_PROFILE_LOG
extern bool saveProfileLog;
#endif // ENABLE_PROFILE_LOG

void mainLoop() { 
#if ENABLE_PROFILE_LOG
    if (saveProfileLog) {
        LogCore::ProfileLog::get().initProfile(renderState->imageName, "end", 2, renderState->iterations - 2, 1, std::ios_base::out);
        std::cout << "Init log profile [" << renderState->imageName << "]" << std::endl;
    }
#endif // ENABLE_PROFILE_LOG
    double fps = 0;
    double timebase = 0;
    int frame = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (!paused) {
            runCuda();
            frame++;
        }

        double time = glfwGetTime();

        if (time - timebase > 1.0) {
            fps = frame / (time - timebase);
            timebase = time;
            frame = 0;
        }

#if ENABLE_PROFILE_LOG
        if (!paused && saveProfileLog) {
            LogCore::ProfileLog::get().step(fps, time * 1000.);
        }
#endif // ENABLE_PROFILE_LOG

        std::string postprocessStr = "PP: ";
        for (size_t i = 0; i < scene->postprocesses.size(); ++i) {
            if (scene->postprocesses[i].second) {
                postprocessStr += std::to_string(i);
            }
        }

#if ENABLE_PROFILE_LOG
        std::string fpsStr;
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << fps;
        ss >> fpsStr;
#endif // ENABLE_PROFILE_LOG
        std::string title = "CIS565 Path Tracer | "
#if ENABLE_PROFILE_LOG
            + std::string(" FPS: ") + fpsStr + " | "
#endif // ENABLE_PROFILE_LOG
            + utilityCore::convertIntToString(renderState->traceDepth) + " Depths" " | "
            + (renderState->recordDepth < 0 ? " All Bounce" " | " : (utilityCore::convertIntToString(renderState->recordDepth) + " Bounce or Upper" " | "))
            + postprocessStr + " | "
            + utilityCore::convertIntToString(iteration) + " Iterations"
#if ENABLE_CACHE_FIRST_INTERSECTION
            + (cacheFirstIntersection ? " (Cache1stBounce)" : "")
#endif // ENABLE_CACHE_FIRST_INTERSECTION
            + (paused ? " (Paused)" : "");
        glfwSetWindowTitle(window, title.c_str());

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

        // Draw imgui
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        drawGui(display_w, display_h);

        glfwSwapBuffers(window);
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}

#pragma warning(pop)
