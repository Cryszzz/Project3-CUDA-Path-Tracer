#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <stb_image_write.h>
#include <stb_image.h>

void printvec3(glm::vec3 vec){
    cout<<vec.x<<" "<<vec.y<<" " <<vec.z<<endl;
}

float cmin(float x, float y, float z){
    return fmin(fmin(x,y),z);
}

float cmin(float x, float y, float z,float w){
    return fmin(fmin(x,y),fmin(z,w));
}

float cmax(float x, float y, float z){
    return fmax(fmax(x,y),z);
}

float cmax(float x, float y, float z,float w){
    return fmax(fmax(x,y),fmax(z,w));
}

void printBVH(std::vector<BVHnode> BVH, std::vector<Geom>& geoms){
    cout<<"BVH leaves count "<<BVH.size()<<endl;
    for(int i=0;i<BVH.size();i++){
        cout<< "BVH min ";
        printvec3(BVH[i].min);
        cout<< "BVH max ";
        printvec3(BVH[i].max);
        cout<< "BVH leftchild "<<BVH[i].leftchild<<endl;
        cout<< "BVH rightchild "<<BVH[i].rightchild<<endl;
        cout<< "BVH leaf "<<BVH[i].leaf<<endl;
        if(BVH[i].leaf){
            cout<< "BVH geom data "<<geoms[BVH[i].geom].type<<endl;
        }
        cout<<" "<<endl;
    }
}
void mergeBox(std::vector<BVHnode> &boxes,BVHnode &parent){
    parent.min=boxes[0].min;
    parent.max=boxes[0].max;
    for (const BVHnode &b : boxes) {
        parent.min=glm::vec3(fmin(b.min.x,parent.min.x),fmin(b.min.y,parent.min.y),fmin(b.min.z,parent.min.z));
        parent.max=glm::vec3(fmax(b.max.x,parent.max.x),fmax(b.max.y,parent.max.y),fmax(b.max.z,parent.max.z));
    }
}

int largest_axis(BVHnode &box){
    glm::vec3 middle=box.max-box.min;
    float max=cmax(middle.x,middle.y,middle.z);
    if(max==middle.x){
        return 0;
    }else if(max==middle.y){
        return 1;
    }else{
        return 2;
    }
}
int constructBVH(std::vector<BVHnode> &boxes,std::vector<BVHnode> &node_pool){
    if (boxes.size() == 1) {
        node_pool.push_back(boxes[0]);
        return node_pool.size() - 1;
    }else if(boxes.size() == 2){
        BVHnode parent;
        parent.leaf=false;
        node_pool.push_back(boxes[0]);
        parent.leftchild=node_pool.size() - 1;
        node_pool.push_back(boxes[1]);
        parent.rightchild=node_pool.size() - 1;
        mergeBox(boxes,parent);
        node_pool.push_back(parent);
        return node_pool.size() - 1;
    }

    BVHnode big_box;
    big_box.leaf=false;
    mergeBox(boxes,big_box);
    int axis = largest_axis(big_box);
    std::vector<BVHnode> local_boxes = boxes;
    std::sort(local_boxes.begin(), local_boxes.end(),
    [&](const BVHnode &b1, const BVHnode &b2) {
        glm::vec3 center1 = (b1.max + b1.min) / 2.0f;
        glm::vec3 center2 = (b2.max + b2.min) / 2.0f;
        return center1[axis] < center2[axis];
    });
    std::vector<BVHnode> left_boxes(
    local_boxes.begin(),
    local_boxes.begin() + local_boxes.size() / 2);
    std::vector<BVHnode> right_boxes(
    local_boxes.begin() + local_boxes.size() / 2,
    local_boxes.end());


    big_box.leftchild= constructBVH(left_boxes, node_pool);
    big_box.rightchild = constructBVH(right_boxes, node_pool);
    node_pool.push_back(big_box);
    return node_pool.size() - 1;
}

void buildBVH(std::vector<BVHnode> &BVH, std::vector<Geom>& geoms){
    //aabb building
    for(int i=0;i<BVH.size();i++){
        Geom & g=geoms[BVH[i].geom];
        if(g.type==TRIANGLE){
            glm::vec3 v0=glm::vec3(g.transform*glm::vec4(g.tri.vertices[0],1.0f));
            glm::vec3 v1=glm::vec3(g.transform*glm::vec4(g.tri.vertices[1],1.0f));
            glm::vec3 v2=glm::vec3(g.transform*glm::vec4(g.tri.vertices[2],1.0f));
            BVH[i].min=glm::vec3(cmin(v0.x,v1.x,v2.x),cmin(v0.y,v1.y,v2.y),cmin(v0.z,v1.z,v2.z))-glm::vec3(0.001f);
            BVH[i].max=glm::vec3(cmax(v0.x,v1.x,v2.x),cmax(v0.y,v1.y,v2.y),cmax(v0.z,v1.z,v2.z))+glm::vec3(0.001f);
        }else{
            glm::vec3 v0=glm::vec3(g.transform*glm::vec4(-0.5f,0.5f,0.5f,1.0f));
            glm::vec3 v1=glm::vec3(g.transform*glm::vec4(-0.5f,-0.5f,0.5f,1.0f));
            glm::vec3 v2=glm::vec3(g.transform*glm::vec4(0.5f,0.5f,0.5f,1.0f));
            glm::vec3 v3=glm::vec3(g.transform*glm::vec4(0.5f,-0.5f,0.5f,1.0f));
            glm::vec3 v4=glm::vec3(g.transform*glm::vec4(-0.5f,0.5f,-0.5f,1.0f));
            glm::vec3 v5=glm::vec3(g.transform*glm::vec4(-0.5f,-0.5f,-0.5f,1.0f));
            glm::vec3 v6=glm::vec3(g.transform*glm::vec4(0.5f,0.5f,-0.5f,1.0f));
            glm::vec3 v7=glm::vec3(g.transform*glm::vec4(0.5f,-0.5f,-0.5f,1.0f));
            BVH[i].min=glm::vec3(fmin(cmin(v0.x,v1.x,v2.x,v3.x),cmin(v4.x,v5.x,v6.x,v7.x)),fmin(cmin(v0.y,v1.y,v2.y,v3.y),cmin(v4.y,v5.y,v6.y,v7.y)),fmin(cmin(v0.z,v1.z,v2.z,v3.z),cmin(v4.z,v5.z,v6.z,v7.z)))-glm::vec3(0.001f);
            BVH[i].max=glm::vec3(fmax(cmin(v0.x,v1.x,v2.x,v3.x),cmax(v4.x,v5.x,v6.x,v7.x)),fmax(cmax(v0.y,v1.y,v2.y,v3.y),cmax(v4.y,v5.y,v6.y,v7.y)),fmax(cmax(v0.z,v1.z,v2.z,v3.z),cmax(v4.z,v5.z,v6.z,v7.z)))+glm::vec3(0.001f);
        }
        
    }
    cout<<"leaf count "<<BVH.size()<<endl;
    std::vector<BVHnode> pool;
    constructBVH(BVH,pool);
    BVH=pool;
    printvec3(BVH[BVH.size()-1].min);
    printvec3(BVH[BVH.size()-1].max);
    cout<<"total count "<<BVH.size()<<endl;

    //printBVH(BVH,geoms);
}

// Load Texture
MyTexture Scene::loadTexture(const std::string& textureFile) {
    MyTexture texture;
    int width, height, numComponents;

    // Load image data
    unsigned char* data = stbi_load(textureFile.c_str(), &width, &height, &numComponents, 0);

    if (data!=NULL) {
        texture.width = width;
        texture.height = height;
        texture.numComponents = 4; // Force RGBA for consistency
        texture.size = width * height * 4;

        for (int i = 0; i < width * height; ++i) {
            // Fetch pixel components with padding for missing channels
            float r = (numComponents > 0) ? data[i * numComponents + 0] / 255.0f : 0.0f;
            float g = (numComponents > 1) ? data[i * numComponents + 1] / 255.0f : 0.0f;
            float b = (numComponents > 2) ? data[i * numComponents + 2] / 255.0f : 0.0f;
            float a = (numComponents > 3) ? data[i * numComponents + 3] / 255.0f : 1.0f;

            // Push padded pixel to texture data
            texture.data.push_back(glm::vec4(r, g, b, a));
        }

        stbi_image_free(data);
    } else {
        std::cerr << "Failed to load texture: " << textureFile << "\nReason: " 
                << stbi_failure_reason() << std::endl;
    }

    return texture;

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
            } else if(strcmp(tokens[0].c_str(), "BACKGROUND_COLOR") == 0){
                backColor=glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                cout<< "background color";
                printvec3(backColor);
            }
        }
    }
    buildBVH(BVH,geoms);
    cout<< "light count " <<Lights.size()<<endl;
    cout<< "light area " <<LightArea.size()<<endl;
    for(int i=0;i<LightArea.size();i++){
        LightArea[i]*=geoms[Lights[i*2]].scale.x*geoms[Lights[i*2]].scale.y*geoms[Lights[i*2]].scale.z;
        cout<< LightArea[i] <<endl;
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
        BVHnode node;

        //load object type
        node.leaf=true;
        int starting_index=geoms.size();
        int tri_size=starting_index;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
                node.geom=geoms.size();;
                BVH.push_back(node);
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
                node.geom=geoms.size();;
                BVH.push_back(node);
            } else if ( strcmp(line.c_str(), "mesh") == 0){
                cout << "Creating new mesh..." << endl;
                newGeom.type = MESH;
                //code referenced from https://github.com/tinyobjloader/tinyobjloader/blob/release/examples
                utilityCore::safeGetline(fp_in, line);
                if (!line.empty() && fp_in.good()) {
                    tinyobj::attrib_t attrib;
                    std::vector<tinyobj::shape_t> shapes;
                    std::vector<tinyobj::material_t> meshmaterials;
                    std::string warn, err;
                    int materialStartIdx = materials.size();
                    std::string objPrefix = line.substr(0, line.find_last_of("/") + 1);
                    std::cout << "OBJ file prefix: " << objPrefix << std::endl;

                    bool ret = tinyobj::LoadObj(&attrib, &shapes, &meshmaterials, &warn, &err, line.c_str(), objPrefix.c_str());

                    if (!warn.empty()) std::cout << "Warning: " << warn << std::endl;
                    if (!err.empty()) std::cerr << "Error: " << err << std::endl;

                    if (!ret) {
                        std::cerr << "Failed to load OBJ: " << line << std::endl;
                        return 0;
                    }

                    if (shapes.empty()) {
                        std::cerr << "Error: No shapes found in OBJ file." << std::endl;
                        return 0;
                    }

                    // Load materials
                    for (const auto& material : meshmaterials) {
                        Material mat;
                        mat.color = glm::vec3(material.diffuse[0], material.diffuse[1], material.diffuse[2]);
                        mat.specular.color = glm::vec3(material.specular[0], material.specular[1], material.specular[2]);
                        mat.specular.exponent = material.shininess;
                        mat.hasReflective = (material.illum >= 3) ? 1.0f : 0.0f;
                        mat.hasRefractive = glm::length(glm::vec3(material.transmittance[0], material.transmittance[1], material.transmittance[2]));
                        mat.indexOfRefraction = material.ior > 0 ? material.ior : 1.0f;
                        mat.emittance = glm::length(glm::vec3(material.emission[0], material.emission[1], material.emission[2]));

                        // Handle diffuse texture
                        if (!material.diffuse_texname.empty()) {
                            MyTexture diffuseTexture = loadTexture(objPrefix + material.diffuse_texname);
                            mat.texture.diffuseID = textures.size();
                            textures.push_back(diffuseTexture);
                        } else {
                            mat.texture.diffuseID = -1;
                        }

                        // Handle normal texture
                        if (!material.normal_texname.empty()) {
                            MyTexture normalTexture = loadTexture(objPrefix + material.normal_texname);
                            mat.texture.normalID = textures.size();
                            textures.push_back(normalTexture);
                        } else {
                            mat.texture.normalID = -1;
                        }

                        if (mat.emittance > 0.0f) {
                            lightmat.push_back(materials.size());
                        }

                        materials.push_back(mat);
                    }

                    // Load shapes
                    for (const auto& shape : shapes) {
                        int meshStart = geoms.size();
                        float totalArea = 0.0f;

                        for (size_t i = 0; i < shape.mesh.indices.size() / 3; i++) {
                            Geom triGeom;
                            Triangle newTri;
                            BVHnode trinode;
                            trinode.leaf = true;
                            triGeom.type = TRIANGLE;
                            triGeom.materialid = shape.mesh.material_ids[i] + materialStartIdx;

                            for (int k = 0; k < 3; k++) {
                                int vertexIndex = shape.mesh.indices[3 * i + k].vertex_index;
                                int normalIndex = shape.mesh.indices[3 * i + k].normal_index;
                                int texcoordIndex = shape.mesh.indices[3 * i + k].texcoord_index;

                                if (vertexIndex != -1) {
                                    newTri.vertices[k] = glm::vec3(
                                        attrib.vertices[3 * vertexIndex + 0],
                                        attrib.vertices[3 * vertexIndex + 1],
                                        attrib.vertices[3 * vertexIndex + 2]
                                    );
                                }

                                if (normalIndex != -1) {
                                    newTri.normals[k] = glm::vec3(
                                        attrib.normals[3 * normalIndex + 0],
                                        attrib.normals[3 * normalIndex + 1],
                                        attrib.normals[3 * normalIndex + 2]
                                    );
                                }

                                if (texcoordIndex != -1) {
                                    newTri.uvs[k] = glm::vec2(
                                        attrib.texcoords[2 * texcoordIndex + 0],
                                        attrib.texcoords[2 * texcoordIndex + 1]
                                    );
                                }
                            }

                            // Compute additional triangle properties
                            auto e1 = newTri.vertices[0] - newTri.vertices[2];
                            auto e2 = newTri.vertices[1] - newTri.vertices[2];
                            auto uv0 = newTri.uvs[0], uv1 = newTri.uvs[1], uv2 = newTri.uvs[2];

                            float constant = 1.0f / ((uv0.x - uv2.x) * (uv1.y - uv2.y) - (uv0.y - uv2.y) * (uv1.x - uv2.x));
                            newTri.dpdu = glm::normalize(((uv1.y - uv2.y) * e1 - (uv0.y - uv2.y) * e2) * constant);
                            newTri.dpdv = glm::normalize((-(uv1.x - uv2.x) * e1 + (uv0.x - uv2.x) * e2) * constant);
                            newTri.g_norm = glm::normalize(newTri.normals[0] + newTri.normals[1] + newTri.normals[2]);

                            newTri.size = glm::length(glm::cross(e1, e2));
                            totalArea += newTri.size;

                            triGeom.tri = newTri;
                            trinode.geom = geoms.size();
                            geoms.push_back(triGeom);
                            BVH.push_back(trinode);
                            tri_size++;
                        }

                        int matid = shape.mesh.material_ids[0] + materialStartIdx;
                        if (std::find(lightmat.begin(), lightmat.end(), matid) != lightmat.end()) {
                            int meshEnd = meshStart + shape.mesh.indices.size() / 3;
                            Lights.push_back(meshStart);
                            Lights.push_back(meshEnd);
                            LightArea.push_back(totalArea);
                        }
                    }
                }
            }
        }
        //link material
       // cout << "starting index" <<starting_index<< endl;
        //cout << "end index" <<tri_size<< endl;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            
            vector<string> tokens = utilityCore::tokenizeString(line);
            int materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << materialid << "..." << endl;
            bool isLight=std::find(lightmat.begin(), lightmat.end(), materialid) != lightmat.end();
            if(newGeom.type == MESH){
                if(materialid!=-1){
                    
                        
                    cout<<"not here"<<endl;
                    float LS=0.0f;
                    for(int i=starting_index;i<tri_size;i++){
                        geoms[i].materialid=materialid;
                        //g.materialid=materialid;
                        LS+=glm::length(glm::cross(geoms[i].tri.vertices[1]-geoms[i].tri.vertices[0],geoms[i].tri.vertices[2]-geoms[i].tri.vertices[0]));
                    }
                    if(isLight) {
                        Lights.push_back(starting_index);
                        Lights.push_back(tri_size);
                        LightArea.push_back(LS);
                    }
                }
            }else{
                newGeom.materialid=materialid;
                if(isLight){
                    Lights.push_back(geoms.size());
                    Lights.push_back(geoms.size()+1);
                    if(newGeom.type==SPHERE)
                        LightArea.push_back(glm::pi<float>());
                    else
                        LightArea.push_back(6.0f);
                }
            }
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            
            //load tranformations
            glm::vec3 translation;
            glm::vec3 rotation;
            glm::vec3 scale;
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                if(newGeom.type == MESH){
                    for(int i=starting_index;i<tri_size;i++){
                        geoms[i].translation=translation;
                    }
                }else{
                    newGeom.translation=translation;
                }
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                if(newGeom.type == MESH){
                    for(int i=starting_index;i<tri_size;i++){
                        geoms[i].rotation=rotation;
                    }
                }else{
                    newGeom.rotation=rotation;
                }
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                if(newGeom.type == MESH){
                    for(int i=starting_index;i<tri_size;i++){
                        geoms[i].scale=scale;
                    }
                }else{
                    newGeom.scale=scale;
                }
            }
            
            
            utilityCore::safeGetline(fp_in, line);
        }

        if(newGeom.type == MESH){
            for(int i=starting_index;i<tri_size;i++){
                geoms[i].transform = utilityCore::buildTransformationMatrix(geoms[i].translation, geoms[i].rotation, geoms[i].scale);
                geoms[i].inverseTransform = glm::inverse(geoms[i].transform);
                geoms[i].invTranspose = glm::inverseTranspose(geoms[i].transform);
                
            }
        }else{
            newGeom.transform = utilityCore::buildTransformationMatrix(newGeom.translation, newGeom.rotation, newGeom.scale);
            newGeom.inverseTransform = glm::inverse(newGeom.transform);
            newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        }
        
        
        if(newGeom.type != MESH){
            geoms.push_back(newGeom);
        }

        
            
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
    cout<< state.traceDepth<<endl;

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
        newMaterial.texture.diffuseID=-1;
        newMaterial.texture.normalID=-1;

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
                if(newMaterial.emittance>0)
                    lightmat.push_back(materials.size());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}
