#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <memory>  // c++11

#define TINYGLTF_IMPLEMENTATION
//#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT

#include <tiny_gltf.h>


static std::string GetFilePathExtension(const std::string& FileName) {
  if (FileName.find_last_of(".") != std::string::npos)
    return FileName.substr(FileName.find_last_of(".") + 1);
  return "";
}


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
                loadMaterial(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "GLTF") == 0) {
                loadGLTF(tokens[1], 3.0);
                cout << " " << endl;
            }
        }
    }
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
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
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
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
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
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

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
            } else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            } else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

int Scene::loadGLTF(const std::string& filename, float scale) {
  /*
  This load function is based on tinygltf's example
  */
  tinygltf::Model model;
  tinygltf::TinyGLTF loader;
  std::string err;
  std::string warn;
  const std::string ext = GetFilePathExtension(filename);

  bool ret = false;
  if (ext.compare("glb") == 0) {
    // assume binary glTF.
    ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename.c_str());
  }
  else {
    // assume ascii glTF.
    ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename.c_str());
  }

  if (!warn.empty()) {
    std::cout << "glTF parse warning: " << warn << std::endl;
  }

  if (!err.empty()) {
    std::cerr << "glTF parse error: " << err << std::endl;
  }
  if (!ret) {
    std::cerr << "Failed to load glTF: " << filename << std::endl;
    return false;
  }

  std::cout << "loaded glTF file has:\n"
    << model.accessors.size() << " accessors\n"
    << model.animations.size() << " animations\n"
    << model.buffers.size() << " buffers\n"
    << model.bufferViews.size() << " bufferViews\n"
    << model.materials.size() << " materials\n"
    << model.meshes.size() << " meshes\n"
    << model.nodes.size() << " nodes\n"
    << model.textures.size() << " textures\n"
    << model.images.size() << " images\n"
    << model.skins.size() << " skins\n"
    << model.samplers.size() << " samplers\n"
    << model.cameras.size() << " cameras\n"
    << model.scenes.size() << " scenes\n"
    << model.lights.size() << " lights\n";

  // Store node as geometries
  for (const tinygltf::Node& gltfNode : model.nodes) {
    Geom newGeom;
    newGeom.type = MESH;
    newGeom.meshid = gltfNode.mesh;
    newGeom.materialid = 1;
    geoms.push_back(newGeom);
  }

  // Get all meshes
  for (const tinygltf::Mesh& gltfMesh : model.meshes) {
    std::cout << "Current mesh has " << gltfMesh.primitives.size()
      << " primitives:\n";

    // Create a mesh object
    Mesh loadedMesh;

    // For each primitive
    for (const tinygltf::Primitive& meshPrimitive : gltfMesh.primitives) {      
      const tinygltf::Accessor& indicesAccessor = model.accessors[meshPrimitive.indices];
      const tinygltf::BufferView& bufferView = model.bufferViews[indicesAccessor.bufferView];
      const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
      const uint16_t* indices = reinterpret_cast<const uint16_t*>(&buffer.data[bufferView.byteOffset + indicesAccessor.byteOffset]);

      const auto byteStride = indicesAccessor.ByteStride(bufferView);
      const size_t count = indicesAccessor.count;
      loadedMesh.count = count;

      // Load indices
      loadedMesh.i_offset = mesh_indices.size();
      for (int i = 0; i < indicesAccessor.count; ++i) {
        mesh_indices.push_back(indices[i]);
      }
      std::cout << '\n';

      switch (meshPrimitive.mode) {
        case TINYGLTF_MODE_TRIANGLES:
        {
          std::cout << "TRIANGLES\n";

          for (const auto & attribute : meshPrimitive.attributes) {
            const tinygltf::Accessor& attribAccessor = model.accessors[attribute.second];
            const tinygltf::BufferView& bufferView = model.bufferViews[attribAccessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
            const float* data = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + indicesAccessor.byteOffset]);
            const auto byte_stride = attribAccessor.ByteStride(bufferView);

            std::cout << "current attribute has count " << count
              << " and stride " << byte_stride << " bytes\n";

            std::cout << "attribute string is : " << attribute.first << '\n';
            if (attribute.first == "POSITION") {
              std::cout << "found position attribute\n";

              // get the position min/max for computing the boundingbox
              loadedMesh.bbox_min.x = attribAccessor.minValues[0];
              loadedMesh.bbox_min.y = attribAccessor.minValues[1];
              loadedMesh.bbox_min.z = attribAccessor.minValues[2];
              loadedMesh.bbox_max.x = attribAccessor.maxValues[0];
              loadedMesh.bbox_max.y = attribAccessor.maxValues[1];
              loadedMesh.bbox_max.z = attribAccessor.maxValues[2];

              // TODO: Allow custom xform

              loadedMesh.bbox_max *= scale;
              loadedMesh.bbox_min *= scale;

              // Store mesh vertices
              loadedMesh.v_offset = mesh_vertices.size();
              for (int i = 0; i < attribAccessor.count; i++) {
                glm::vec3 v;
                v.x = data[i * 3 + 0];
                v.y = data[i * 3 + 1];
                v.z = data[i * 3 + 2];
                mesh_vertices.push_back(v*scale + glm::vec3(0,1,0));
              }
            }
            else if (attribute.first == "NORMAL") {
              std::cout << "found normal attribute\n";

              loadedMesh.n_offset = mesh_normals.size();

              // IMPORTANT: We need to reorder normals (and texture
              // coordinates into "facevarying" order) for each face
              // 
              // For each triangle :
              for (int i = 0; i < loadedMesh.count; i+=3) {
                // get the i'th triange's indexes
                int f0 = indices[i + 0];
                int f1 = indices[i + 1];
                int f2 = indices[i + 2];

                // get the 3 normal vectors for that face
                glm::vec3 n0, n1, n2;
                n0.x = data[3*f0 + 0];
                n0.y = data[3*f0 + 1];
                n0.z = data[3*f0 + 2];
                n1.x = data[3*f1 + 0];
                n1.y = data[3*f1 + 1];
                n1.z = data[3*f1 + 2];
                n2.x = data[3*f2 + 0];
                n2.y = data[3*f2 + 1];
                n2.z = data[3*f2 + 2];

                // Put them in the array in the correct order
                mesh_normals.push_back(n0);
                mesh_normals.push_back(n1);
                mesh_normals.push_back(n2);
              }
            }
          }
        }
        default:
          std::cerr << "primitive mode not implemented";
          break;
      }
    }

    meshes.push_back(loadedMesh);
    ret = true;
  }

  return ret;
}
