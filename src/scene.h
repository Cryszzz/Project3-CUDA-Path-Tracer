#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "image.h"
using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    MyTexture loadTexture(const std::string& textureFile);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<BVHnode> BVH;
    std::vector<int> lightmat;
    std::vector<int> Lights;
    std::vector<float> LightArea;
    std::vector<Material> materials;
    std::vector<MyTexture> textures;
    MyTexture envmap;
    glm::vec4 backColor=glm::vec4(0.0f);
    RenderState state;
};
