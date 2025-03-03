#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "image.h"
#include "glm/glm.hpp"
#include "utilities.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    MESH,
    TRIANGLE
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Triangle {
    glm::vec3 vertices[3];
    glm::vec3 normals[3];
    glm::vec2 uvs[3];
    glm::vec3 g_norm;
    int matID;
    float size;
    glm::vec3 dpdu;
    glm::vec3 dpdv;
};

struct Geom {
    enum GeomType type;
    int materialid;
    Triangle tri;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct BVHnode {
    glm::vec3 min;
    glm::vec3 max;
    int leftchild;
    int rightchild;
    bool leaf;
    int geom;
};

struct MyTexture{
	int width;
	int height;
	int numComponents;
    int size;
	std::vector<glm::vec4> data; 
};

struct TextID{
    int diffuseID;
    int normalID;
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
    TextID texture;
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    glm::vec3 throughput;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  glm::vec2 uv;
  int materialId;
  bool outside;
  glm::vec3 dpdu;
  glm::vec3 dpdv;
};
