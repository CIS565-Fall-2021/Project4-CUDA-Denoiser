#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <chrono>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "tiny_gltf.h"
#include "tiny_obj_loader.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    int loadCameraGLTF(tinygltf::Model);
    int loadMaterialOBJ(tinyobj::ObjReader reader);
    int loadCameraOBJ(tinyobj::ObjReader reader);
    int loadGeomOBJ(tinyobj::ObjReader reader);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    glm::vec3 obj_mins;
    glm::vec3 obj_maxs;

    std::vector<Triangle> triangles;
    int disp_idx;

    bool denoise;
    float dn_filterSize;
    float dn_colorWeight;
    float dn_normalWeight;
    float dn_positionWeight;

    std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::duration<int64_t, std::ratio<1, 1000000000>>> start_t;
    std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::duration<int64_t, std::ratio<1, 1000000000>>> stop_t;
    bool printed_t;


};
