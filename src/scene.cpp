#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <random>

#define LOAD_GEOM_AND_MAT_FROM_FILE 1

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterialFromFile(tokens[1]);
                cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeomFromFile(tokens[1]);
                cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }
#if !LOAD_GEOM_AND_MAT_FROM_FILE
    loadGeoAndMat();
#endif
}

inline static float randomFloat() {
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    static std::mt19937 generator;
    return distribution(generator);
}

// Returns a random real in [min,max).
inline static float randomFloat(float min, float max) {
    return min + (max - min) * randomFloat();
}

// return a vector where each component is a random number from [0,1)
inline static glm::vec3 randomVec3()
{
    return glm::vec3(randomFloat(), randomFloat(), randomFloat());
}

// return a vector where each component is a random number from [min,max)
inline static glm::vec3 randomVec3(float min, float max)
{
    return glm::vec3(randomFloat(min, max), randomFloat(min, max), randomFloat(min, max));
}

void Scene::loadGeoAndMat()
{
    int materialCounter = 0;
    int geoCounter = 0;

    Material groundMaterial;
    groundMaterial.color = glm::vec3(0.5f);
    groundMaterial.specular.exponent = 0;
    groundMaterial.specular.color = glm::vec3(0.f);
    groundMaterial.hasReflective = 0;
    groundMaterial.hasRefractive = 0;
    groundMaterial.indexOfRefraction = 0;
    groundMaterial.emittance = 0;
    groundMaterial.fuzziness = 0;
    materials.push_back(groundMaterial);

    Geom groundSphere;
    groundSphere.type = SPHERE;
    groundSphere.materialid = materialCounter++;
    groundSphere.translation = glm::vec3(0.f, -1000.f, 0.f);
    groundSphere.rotation = glm::vec3(0.f);
    groundSphere.scale = glm::vec3(2000.f);
    groundSphere.transform = utilityCore::buildTransformationMatrix(
        groundSphere.translation, groundSphere.rotation, groundSphere.scale);
    groundSphere.inverseTransform = glm::inverse(groundSphere.transform);
    groundSphere.invTranspose = glm::inverseTranspose(groundSphere.transform);
    geoms.push_back(groundSphere);

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto chooseMat = randomFloat();
            glm::vec3 center(a + 0.9f * randomFloat(), 0.2f, b + 0.9f * randomFloat());

            if (glm::length(center - glm::vec3(4.0f, 0.2f, 0.0f)) > 0.9f)
            {
                Material sphereMaterial;

                if (chooseMat < 0.8f) {
                    // diffuse
                    auto albedo = randomVec3() * randomVec3();
                    sphereMaterial.color = albedo;
                    sphereMaterial.specular.exponent = 0;
                    sphereMaterial.specular.color = glm::vec3(0.f);
                    sphereMaterial.hasReflective = 0;
                    sphereMaterial.hasRefractive = 0;
                    sphereMaterial.indexOfRefraction = 0;
                    sphereMaterial.emittance = 0;
                    sphereMaterial.fuzziness = 0;
                }
                else if (chooseMat < 0.95f) {
                    // metal
                    auto albedo = randomVec3(0.5f, 1.0f);
                    auto fuzz = randomFloat(0.0f, 0.5f);
                    sphereMaterial.color = albedo;
                    sphereMaterial.specular.exponent = 0;
                    sphereMaterial.specular.color = albedo;
                    sphereMaterial.hasReflective = 1;
                    sphereMaterial.hasRefractive = 0;
                    sphereMaterial.indexOfRefraction = 0;
                    sphereMaterial.emittance = 0;
                    sphereMaterial.fuzziness = fuzz;
                }
                else {
                    // glass
                    sphereMaterial.color = glm::vec3(1.0f);
                    sphereMaterial.specular.exponent = 0;
                    sphereMaterial.specular.color = glm::vec3(0.f);
                    sphereMaterial.hasReflective = 0;
                    sphereMaterial.hasRefractive = 1;
                    sphereMaterial.indexOfRefraction = 1.5;
                    sphereMaterial.emittance = 0;
                    sphereMaterial.fuzziness = 0;
                }

                materials.push_back(sphereMaterial);

                Geom s1;
                s1.type = SPHERE;
                s1.materialid = materialCounter++;
                s1.translation = center;
                s1.rotation = glm::vec3(0.f);
                s1.scale = glm::vec3(.4f);
                s1.transform = utilityCore::buildTransformationMatrix(
                    s1.translation, s1.rotation, s1.scale);
                s1.inverseTransform = glm::inverse(s1.transform);
                s1.invTranspose = glm::inverseTranspose(s1.transform);
                geoms.push_back(s1);
            }
        }
    }


    Material dielectric;
    dielectric.color = glm::vec3(1.0f);
    dielectric.specular.exponent = 0;
    dielectric.specular.color = glm::vec3(0.f);
    dielectric.hasReflective = 0;
    dielectric.hasRefractive = 1;
    dielectric.indexOfRefraction = 1.5;
    dielectric.emittance = 0;
    dielectric.fuzziness = 0;
    materials.push_back(dielectric);

    Geom s1;
    s1.type = SPHERE;
    s1.materialid = materialCounter++;
    s1.translation = glm::vec3(0.f, 1.f, 0.f);
    s1.rotation = glm::vec3(0.f);
    s1.scale = glm::vec3(2.f);
    s1.transform = utilityCore::buildTransformationMatrix(
        s1.translation, s1.rotation, s1.scale);
    s1.inverseTransform = glm::inverse(s1.transform);
    s1.invTranspose = glm::inverseTranspose(s1.transform);
    geoms.push_back(s1);

    Material lambertian;
    lambertian.color = glm::vec3(.4f, .2f, .1f);
    lambertian.specular.exponent = 0;
    lambertian.specular.color = glm::vec3(0.f);
    lambertian.hasReflective = 0;
    lambertian.hasRefractive = 0;
    lambertian.indexOfRefraction = 0;
    lambertian.emittance = 0;
    lambertian.fuzziness = 0;
    materials.push_back(lambertian);

    Geom s2;
    s2.type = SPHERE;
    s2.materialid = materialCounter++;
    s2.translation = glm::vec3(-4.f, 1.f, 0.f);
    s2.rotation = glm::vec3(0.f);
    s2.scale = glm::vec3(2.f);
    s2.transform = utilityCore::buildTransformationMatrix(
        s2.translation, s2.rotation, s2.scale);
    s2.inverseTransform = glm::inverse(s2.transform);
    s2.invTranspose = glm::inverseTranspose(s2.transform);
    geoms.push_back(s2);

    Material metal;
    metal.color = glm::vec3(.7f, .6f, .5f);
    metal.specular.exponent = 0;
    metal.specular.color = glm::vec3(.7f, .6f, .5f);
    metal.hasReflective = 1;
    metal.hasRefractive = 0;
    metal.indexOfRefraction = 0;
    metal.emittance = 0;
    metal.fuzziness = 0;
    materials.push_back(metal);

    Geom s3;
    s3.type = SPHERE;
    s3.materialid = materialCounter++;
    s3.translation = glm::vec3(4.f, 1.f, 0.f);
    s3.rotation = glm::vec3(0.f);
    s3.scale = glm::vec3(2.f);
    s3.transform = utilityCore::buildTransformationMatrix(
        s3.translation, s3.rotation, s3.scale);
    s3.inverseTransform = glm::inverse(s3.transform);
    s3.invTranspose = glm::inverseTranspose(s3.transform);
    geoms.push_back(s3);
}


int Scene::loadGeomFromFile(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    }
    else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            }
            else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
        return 1;
    }
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState& state = this->state;
    Camera& camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterialFromFile(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    }
    else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        // Adding a new material property needs two modifications: 
        // 1) i < (value + 1)
        // 2) else if statement

        //load static properties
        for (int i = 0; i < 8; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.color = color;
            }
            else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            }
            else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            }
            // Added fuzziness
            else if (strcmp(tokens[0].c_str(), "FUZZ") == 0) {
                newMaterial.fuzziness = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}
