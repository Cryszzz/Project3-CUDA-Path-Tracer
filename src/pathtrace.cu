#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/count.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "OpenImageDenoise/oidn.hpp"

#define SHARC_ENABLE_64_BIT_ATOMICS 1
#define HASH_GRID_ENABLE_64_BIT_ATOMICS 1
#define SHARC_UPDATE 1
#define SHARC_QUERY 1
#define ENABLE_CACHE 1 //SHARC ENABLE CACHE
#include "SHARC/SharcCommon.h"
#define RussianRoulette 0
#define ERRORCHECK 1
#define DENOISE 1
#define STACKSIZE 16384 //262144

__host__ __device__ inline float3 glmToFloat3(const glm::vec3& vec) {
    return make_float3(vec.x, vec.y, vec.z);
}

__host__ __device__ inline glm::vec3 float3ToGlm(const float3& vec) {
    return glm::vec3(vec.x, vec.y, vec.z);
}

__host__ __device__
float computeLuminance(const glm::vec3& color) {
    return 0.2126f * color.r + 0.7152f * color.g + 0.0722f * color.b;
}

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
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
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static size_t bufferSize = (1 << 22); // Example: 2^22 entries
static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static int *dev_keys;
static int *dev_values;
static PathSegment* dev_pathR;
static ShadeableIntersection* dev_intersectionsR;
static PathSegment* finalbuffer;
static ShadeableIntersection* firstBounce = NULL;
static PathSegment* firstBounceP=NULL;
static BVHnode* dev_tree=NULL;
static cudaTextureObject_t* dev_texture_objects = NULL;
static cudaTextureObject_t dev_env = NULL;
static int textureSize=0;
static int* dev_lights=NULL;
static float* dev_lights_area=NULL;
// SHaRC state and buffers
static SharcState sharcState;
static uint4* dev_voxelDataBuffer;
static uint4* dev_voxelDataBufferPrev;
static uint64_t* dev_hashEntriesBuffer;
static uint* dev_copyOffsetBuffer;
static glm::vec3* dev_normalImage;
static glm::vec3* dev_albedoImage;

// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceInit(Scene* scene) {
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
	cudaMalloc(&dev_normalImage, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_normalImage, 0, pixelcount * sizeof(glm::vec3));
	cudaMalloc(&dev_albedoImage, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_albedoImage, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_tree, scene->BVH.size() * sizeof(BVHnode));
	cudaMemcpy(dev_tree, scene->BVH.data(), scene->BVH.size() * sizeof(BVHnode), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	//textimgcnt = scene->imgtextwh.size();
	//cudaMemcpy(textimgcnt, ,sizeof(int), cudaMemcpyHostToDevice);

	if(scene->envmap.data.size()>0){

		const MyTexture& texture = scene->envmap;
		cudaArray_t dev_envtexture;
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
		cudaMallocArray(&dev_envtexture, &channelDesc, scene->envmap.width, scene->envmap.height);
		cudaMemcpy2DToArray(dev_envtexture, 0, 0, texture.data.data(), texture.width * sizeof(float4), texture.width * sizeof(float4), texture.height, cudaMemcpyHostToDevice);

		// Create texture object
		cudaResourceDesc resDesc = {};
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = dev_envtexture;

		cudaTextureDesc texDesc = {};
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 1;

		cudaCreateTextureObject(&dev_env, &resDesc, &texDesc, nullptr);
	}

    // Allocate and copy textures
	cudaMalloc(&dev_texture_objects, scene->textures.size() * sizeof(cudaTextureObject_t));
	textureSize = scene->textures.size();
    for (int i = 0; i < scene->textures.size(); ++i) {
		const MyTexture& texture = scene->textures[i];
        cudaArray_t dev_texture;
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
        cudaMallocArray(&dev_texture, &channelDesc, texture.width, texture.height);
		cudaMemcpy2DToArray(dev_texture, 0, 0, texture.data.data(), texture.width * sizeof(float4), texture.width * sizeof(float4), texture.height, cudaMemcpyHostToDevice);

        // Create texture object
        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = dev_texture;

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;

        cudaTextureObject_t texObj = 0;
        cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

		cudaMemcpy(dev_texture_objects + i, &texObj, sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
    }


	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	
	// TODO: initialize any extra device memeory you need

	cudaMalloc(&dev_lights, scene->Lights.size() * sizeof(int));
	cudaMemcpy(dev_lights,  scene->Lights.data(), scene->Lights.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&dev_lights_area, scene->LightArea.size() * sizeof(int));
	cudaMemcpy(dev_lights_area,  scene->LightArea.data(), scene->LightArea.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dev_keys, pixelcount * sizeof(int));
	cudaMalloc((void**)&dev_values, pixelcount * sizeof(int));
	cudaMalloc((void**)&finalbuffer, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_intersectionsR, pixelcount * sizeof(ShadeableIntersection));
	cudaMalloc(&firstBounce, pixelcount  * sizeof(ShadeableIntersection));
	cudaMalloc(&firstBounceP, pixelcount  * sizeof(PathSegment));
	cudaMalloc(&dev_pathR, pixelcount * sizeof(PathSegment));

	// SHaRC buffer allocations
	
	cudaMalloc(&dev_voxelDataBuffer, bufferSize * sizeof(uint4));
	cudaMalloc(&dev_voxelDataBufferPrev, bufferSize * sizeof(uint4));
	cudaMalloc(&dev_hashEntriesBuffer, bufferSize * sizeof(uint64_t));
	cudaMalloc(&dev_copyOffsetBuffer, bufferSize * sizeof(uint));
	cudaMemset(dev_voxelDataBuffer, 0, bufferSize * sizeof(uint4));
	cudaMemset(dev_voxelDataBufferPrev, 0, bufferSize * sizeof(uint4));
	cudaMemset(dev_hashEntriesBuffer, 0, bufferSize * sizeof(uint64_t));
	cudaMemset(dev_copyOffsetBuffer, 0, bufferSize * sizeof(uint));

	// Initialize SHaRC parameters
	sharcState.gridParameters.cameraPosition = make_float3(cam.position.x, cam.position.y, cam.position.z);
	sharcState.gridParameters.cameraPositionPrev = make_float3(cam.position.x, cam.position.y, cam.position.z);
	sharcState.gridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
	sharcState.gridParameters.sceneScale = 50.0f;
	sharcState.hashMapData.capacity = bufferSize;
	sharcState.hashMapData.hashEntriesBuffer = dev_hashEntriesBuffer;
	sharcState.voxelDataBuffer = dev_voxelDataBuffer;
	sharcState.voxelDataBufferPrev = dev_voxelDataBufferPrev;
	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_values);
	cudaFree(dev_pathR);
	cudaFree(dev_intersectionsR);
	cudaFree(firstBounce);
	cudaFree(firstBounceP);
	cudaFree(dev_keys);
	cudaFree(finalbuffer);
	if (dev_texture_objects != NULL) {
        for (int i = 0; i < textureSize; ++i) {
			cudaTextureObject_t texObj;
			cudaMemcpy(&texObj, dev_texture_objects + i, sizeof(cudaTextureObject_t), cudaMemcpyDeviceToHost);
			cudaDestroyTextureObject(texObj);
        }
    }
	if(dev_env!=NULL){
		cudaDestroyTextureObject(dev_env);
	}
	//cudaDestroyTextureObject(dev_env);
	cudaFree(dev_texture_objects);
	cudaFree(dev_normalImage);
	cudaFree(dev_albedoImage);

	// Free SHaRC buffers
	cudaFree(dev_voxelDataBuffer);
	cudaFree(dev_voxelDataBufferPrev);
	cudaFree(dev_hashEntriesBuffer);
	cudaFree(dev_copyOffsetBuffer);
	checkCUDAError("pathtraceFree");
}

__device__ void getEnvironmentMapColor(
    PathSegment& pathSegment,
	glm::vec3& backcolor,
	float useBack,
    cudaTextureObject_t enviromentMap) {
	glm::vec3 rayDir = -pathSegment.ray.direction;
    rayDir = glm::normalize(rayDir);
    float u = 0.5f + (atan2(rayDir.z, rayDir.x) / (2.0f * PI));
    float v = 0.5f - (asin(rayDir.y) / PI);
	if (useBack == 0.0f) {
		float4 texColor = tex2D<float4>(enviromentMap, u, v);
		backcolor = glm::vec3(texColor.x, texColor.y, texColor.z);
	}		
    pathSegment.color *= backcolor;
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments,bool antialiasing)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	thrust::uniform_real_distribution<float> u01(0, 1);

	thrust::default_random_engine rng1 = makeSeededRandomEngine(iter, x + (y * cam.resolution.x), 0);
	thrust::default_random_engine rng0 = makeSeededRandomEngine(iter, y + (x * cam.resolution.y), 0);
	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
		segment.throughput=glm::vec3(1.0f);
		// TODO: implement antialiasing by jittering the ray
		//+u01(rng1)-0.5f
		//+u01(rng0)-0.5f
		if(antialiasing){
			segment.ray.direction = glm::normalize(cam.view
				- cam.right * cam.pixelLength.x * ((float)x+u01(rng1)-0.5f- (float)cam.resolution.x * 0.5f)
				- cam.up * cam.pixelLength.y * ((float)y +u01(rng0)-0.5f- (float)cam.resolution.y * 0.5f)
			);
		}else{
			segment.ray.direction = glm::normalize(cam.view
				- cam.right * cam.pixelLength.x * ((float)x- (float)cam.resolution.x * 0.5f)
				- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);
		}
		

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
	
	
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, ShadeableIntersection* intersections
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		glm::vec2 uv=glm::vec2(0.0f);
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;
		bool temp_outside=true;
		glm::vec3 temp_dpdu=glm::vec3(0.0f);
		glm::vec3 temp_dpdv=glm::vec3(0.0f);

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec2 tmp_uv=glm::vec2(0.0f);
		glm::vec3 dpdu=glm::vec3(0.0f);
		glm::vec3 dpdv=glm::vec3(0.0f);

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, temp_outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, temp_outside);
			}else if (geom.type == TRIANGLE){
				t = triIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_uv,temp_dpdu, temp_dpdv, tmp_normal, temp_outside);
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
				uv=tmp_uv;
				outside=temp_outside;
				dpdu=temp_dpdu;
				dpdv=temp_dpdv;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].outside=outside;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].uv = uv;
			intersections[path_index].dpdu = dpdu;
			intersections[path_index].dpdv = dpdv;
		}
	}
}

__global__ void computeIntersectionsBVH(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, BVHnode* tree
	, int tree_size
	, ShadeableIntersection* intersections
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		glm::vec2 uv = glm::vec2(0.0f);
		glm::vec3 dpdu=glm::vec3(0.0f);
		glm::vec3 dpdv=glm::vec3(0.0f);
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;
		bool temp_outside=true;
		int stack[STACKSIZE];
		int stackptr=0;
		int stacksize=0;
		stack[0]=tree_size-1;
		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec2 tmp_uv= glm::vec2(0.0f);
		glm::vec3 temp_dpdu=glm::vec3(0.0f);
		glm::vec3 temp_dpdv=glm::vec3(0.0f);

		// naive parse through global geoms
		while(true){
			BVHnode& node=tree[stack[stackptr]];
			stackptr=(stackptr+1)%STACKSIZE;;
			if(node.leaf){
				Geom& geom = geoms[node.geom];
				if (geom.type == CUBE)
				{
					t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, temp_outside);
				}
				else if (geom.type == SPHERE)
				{
					t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, temp_outside);
				}else if (geom.type == TRIANGLE){
					t = triIntersectionTest(geom, pathSegment.ray, tmp_intersect,tmp_uv,temp_dpdu, temp_dpdv,tmp_normal, temp_outside);
				}
				if (t > 0.0f && t_min > t)
				{
					t_min = t;
					hit_geom_index = geom.materialid;
					intersect_point = tmp_intersect;
					normal = tmp_normal;
					uv= tmp_uv;
					dpdu=temp_dpdu;
					dpdv=temp_dpdv;
					outside=temp_outside;
				}
			}else{
				float dist1=aabbIntersectionTest(tree[node.leftchild], pathSegment.ray, tmp_intersect, tmp_normal, temp_outside);
				if(dist1>=0.0f){
					stacksize=(stacksize+1)%STACKSIZE;;
					stack[stacksize]=node.leftchild;
				}
				if(node.rightchild!=-1){
					float dist2=aabbIntersectionTest(tree[node.rightchild], pathSegment.ray, tmp_intersect, tmp_normal, temp_outside);
					if(dist2>=0.0f){
						stacksize=(stacksize+1)%STACKSIZE;
						stack[stacksize]=node.rightchild;
					}
				}
			}
			if(stackptr==stacksize+1){
				break;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].outside=outside;
			intersections[path_index].materialId = hit_geom_index;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].uv = uv;
			intersections[path_index].dpdu = dpdu;
			intersections[path_index].dpdv = dpdv;
		}
	}
}

void denoiseImage(const std::vector<glm::vec3>* inputImage, 
    const std::vector<glm::vec3>* albedoImage,
	const std::vector<glm::vec3>* normalImage,
    std::vector<glm::vec3>* outputImage, 
    int numPixels, glm::ivec2 camResolution)
{
    const char* errorMessage;
    oidn::DeviceRef device = oidn::newDevice();
	device.commit();

	oidn::BufferRef colorBuf = device.newBuffer(camResolution.x * camResolution.y * 3 * sizeof(float)); // beauty buffer
	oidn::BufferRef albedoBuf = device.newBuffer(camResolution.x * camResolution.y * 3 * sizeof(float)); // albedo buffer
	oidn::BufferRef normalBuf = device.newBuffer(camResolution.x * camResolution.y * 3 * sizeof(float)); // normal buffer


	oidn::FilterRef filter = device.newFilter("RT");
    filter.setImage("color", colorBuf, oidn::Format::Float3, camResolution.x, camResolution.y);
	filter.setImage("albedo", albedoBuf, oidn::Format::Float3, camResolution.x, camResolution.y);
	filter.setImage("normal", normalBuf, oidn::Format::Float3, camResolution.x, camResolution.y);
    filter.setImage("output", colorBuf, oidn::Format::Float3, camResolution.x, camResolution.y);
    filter.set("hdr", true);
    filter.commit();
    if (device.getError(errorMessage) != oidn::Error::None)
        std::cerr << "Error: " << errorMessage << std::endl;

    float* colorPtr = (float*)colorBuf.getData();
	for (int i = 0; i < numPixels; ++i) {
		colorPtr[i * 3] = (*inputImage)[i].x;
		colorPtr[i * 3 + 1] = (*inputImage)[i].y;
		colorPtr[i * 3 + 2] = (*inputImage)[i].z;
	}

	float* albedoPtr = (float*)albedoBuf.getData();
	for (int i = 0; i < numPixels; ++i) {
		albedoPtr[i * 3] = (*albedoImage)[i].x;
		albedoPtr[i * 3 + 1] = (*albedoImage)[i].y;
		albedoPtr[i * 3 + 2] = (*albedoImage)[i].z;
	}

	float* normalPtr = (float*)normalBuf.getData();
	for (int i = 0; i < numPixels; ++i) {
		glm::vec3 normal = glm::normalize((*normalImage)[i]);
		normalPtr[i * 3] = normal.x;
		normalPtr[i * 3 + 1] = normal.y;
		normalPtr[i * 3 + 2] = normal.z;
	}

	filter.execute();


    if (device.getError(errorMessage) != oidn::Error::None)
        std::cerr << "Error: " << errorMessage << std::endl;

    // copy back to outputImage
	colorPtr = (float*)colorBuf.getData();
	for (int i = 0; i < numPixels; ++i) {
		(*outputImage)[i] = glm::vec3(colorPtr[i * 3], colorPtr[i * 3 + 1], colorPtr[i * 3 + 2]);
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
__global__ void shadeFakeMaterial(
	int iter,
	int depth
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, glm::vec3* albedoImage
	, glm::vec3* normalImage
	, Material* materials
	, glm::vec4 backUsage
	, cudaTextureObject_t env
	, cudaTextureObject_t * texts
	, Geom* geoms
	, int* Lights
	, float* LightArea
	, int lightsize
	, int shading
	, SharcState sharcState
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	bool updateCache = depth%2==0;
	if (idx < num_paths)
	{	
		if(depth==0){
			albedoImage[pathSegments[idx].pixelIndex]=materials[idx].color;
			normalImage[pathSegments[idx].pixelIndex]=shadeableIntersections[idx].surfaceNormal;
		}
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
		  // Set up the RNG
		  // LOOK: this is how you use thrust's RNG! Please look at
		  // makeSeededRandomEngine as well.
		  	Ray& ray = pathSegments[idx].ray;
		  	SharcHitData sharcHitData;
			sharcHitData.positionWorld = glmToFloat3(ray.origin + ray.direction * intersection.t);
			sharcHitData.normalWorld = glmToFloat3(intersection.surfaceNormal);
		  	glm::vec3 throughput = glm::vec3(1.0f);
		  	if (ENABLE_CACHE&&!updateCache) {
				uint gridLevel = GetGridLevel(glmToFloat3(ray.origin + ray.direction * intersection.t), sharcState.gridParameters);
                float voxelSize = GetVoxelSize(gridLevel, sharcState.gridParameters);
				bool isValidHit = intersection.t > (voxelSize * sqrt(5.0f));
                float3 cachedRadiance;
                if (isValidHit&& SharcGetCachedRadiance(sharcState, sharcHitData, cachedRadiance, false)) {
					//pathSegments[idx].color = float3ToGlm(SharcDebugBitsOccupancyRadiance(sharcState, sharcHitData));
                    pathSegments[idx].color *= float3ToGlm(cachedRadiance);
                    pathSegments[idx].remainingBounces = 0; // Terminate path
                    return;
                }
            }
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			
			thrust::uniform_real_distribution<float> u01(0, 1);
			int index=((int)(u01(rng)*((float)lightsize/2)));
			int start=Lights[index*2];
			int end=Lights[index*2+1];
			int gidx=u01(rng)*(end-start)+start;
			if (intersection.materialId == 5)
				int test = 1;
			scatterRay(pathSegments[idx],intersection,materials[intersection.materialId],rng,texts,geoms[gidx],LightArea[index],shading,throughput);
			if (ENABLE_CACHE&&pathSegments[idx].remainingBounces>0&&updateCache) {
				if(!SharcUpdateHit(sharcState, sharcHitData, glmToFloat3(pathSegments[idx].color), u01(rng))){
					pathSegments[idx].remainingBounces = 0; // Terminate path
					return;
				};
			}
			pathSegments[idx].throughput*= throughput;
			if(RussianRoulette&&pathSegments[idx].remainingBounces>0){
				if (depth > 3) {
					float q = glm::min(1.0f, computeLuminance(pathSegments[idx].throughput));
					float qt = glm::min(1.0f, computeLuminance(throughput));
					thrust::uniform_real_distribution<float> u01(0, 1);
					float randomValue = u01(rng);
					if (randomValue > q) {
						pathSegments[idx].remainingBounces = 0; // Terminate path
						return;
					}else{
						throughput /= qt;
						pathSegments[idx].throughput/=q;
					}
				}
			}
			if (ENABLE_CACHE&&pathSegments[idx].remainingBounces>0&&updateCache) {
				SharcSetThroughput(sharcState, glmToFloat3(throughput*(0.5f)));
			}
			// If the material indicates that the object was a light, "light" the ray
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			glm::vec3 backColor=glm::vec3(backUsage.x,backUsage.y,backUsage.z);
			getEnvironmentMapColor(pathSegments[idx],backColor,backUsage.w,env);
			pathSegments[idx].remainingBounces=0;
			if (ENABLE_CACHE&&updateCache) {
				SharcUpdateMiss(sharcState,  glmToFloat3(backColor));
			}
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

__global__ void setfinalbuffer(int nPaths, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		iterationPaths[index].pixelIndex=index;
		iterationPaths[index].color=glm::vec3(0.0f);

	}
}

__global__ void materialRemap(int num_paths,
	PathSegment* dev_paths,
	ShadeableIntersection* intersection,
	int *dev_keys,
	int *dev_values){

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < num_paths)
	{
		dev_keys[index]=intersection[index].materialId;
		dev_values[index]=index;
	}
}

__global__ void tonemapKernel(glm::vec3* hdrImage, int width, int height, float gamma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = x + y * width;

    // Fetch HDR color
    glm::vec3 hdrColor = hdrImage[idx];
	float luminance = glm::dot(hdrColor, glm::vec3(0.2126f, 0.7152f, 0.0722f));

	// Apply Reinhard tone mapping to luminance
	float mappedLuminance = luminance / (luminance + 1.0f);

	if (luminance > 0.0f) {
		hdrColor *= (mappedLuminance / luminance);
	}

    hdrColor = glm::pow(hdrColor, glm::vec3(1.0f / gamma));

}



struct is_zero
{
  __host__ __device__
  bool operator()(PathSegment x)
  {
    return x.remainingBounces  == 0;
  }
};

/**
 * Performs scatter on an array. That is, for each element in idata,
 * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
 */
 __global__ void kernScatter(int n, PathSegment* idata,PathSegment* finaldata) {
	// TODO
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) {
		return;
	}
	if (idata[index].remainingBounces == 0) {
		finaldata[idata[index].pixelIndex].color=idata[index].color;
	}

}


__global__ void kernReshuffle(int N, int* particleArrayIndices, PathSegment* pos,
	PathSegment* posR, ShadeableIntersection* vel, ShadeableIntersection* velR){
	  int index = threadIdx.x + (blockIdx.x * blockDim.x);
	  if (index >= N) {
		return;
	  }
	  posR[index]=pos[particleArrayIndices[index]];
	  velR[index]=vel[particleArrayIndices[index]];
  
}

__global__ void sharcCompactionKernel(
    uint64_t* hashEntriesBuffer,
    uint* copyOffsetBuffer,
    uint capacity) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	HashMapData hashMapData;
    hashMapData.capacity = capacity;
    hashMapData.hashEntriesBuffer = hashEntriesBuffer;
	SharcCopyHashEntry(idx, hashMapData, copyOffsetBuffer);
}

__global__ void sharcResolveKernel(
    uint4* voxelDataBuffer,
    uint4* voxelDataBufferPrev,
    uint64_t* hashEntriesBuffer,
    uint* copyOffsetBuffer,
    float3 cameraPosition,
    float3 cameraPositionPrev,
    float sceneScale,
    uint capacity,
    uint accumulationFrameNum,
    uint staleFrameNum) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Resolve entry
    GridParameters gridParameters;
    gridParameters.cameraPosition = cameraPosition;
    gridParameters.cameraPositionPrev = cameraPositionPrev;
    gridParameters.sceneScale = sceneScale;
    gridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;

    HashMapData hashMapData;
    hashMapData.capacity = capacity;
    hashMapData.hashEntriesBuffer = hashEntriesBuffer;

    SharcResolveEntry(
        idx, 
        gridParameters, 
        hashMapData, 
        copyOffsetBuffer, 
        voxelDataBuffer, 
        voxelDataBufferPrev, 
        accumulationFrameNum, 
        staleFrameNum);
}


  /**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
 /*
	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing
	
*/
void pathtraceSortMatWCacheBVH(uchar4* pbo, int frame, int iter,bool Cache, bool Antialiasing, bool sortMat, bool BVH,int shading) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	if (ENABLE_CACHE) {
		auto temp = dev_voxelDataBufferPrev;
		dev_voxelDataBufferPrev = dev_voxelDataBuffer;
		dev_voxelDataBuffer = temp;
		cudaMemset(dev_voxelDataBuffer, 0, bufferSize * sizeof(uint4));
		cudaMemset(dev_voxelDataBufferPrev, 0, bufferSize * sizeof(uint4));
	}
	// 1D block for path tracing
	const int blockSize1d = 128;
	if(Cache){
		if(iter==1)
			generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter,traceDepth, dev_paths,false);
		else
			cudaMemcpy(dev_paths,firstBounceP,pixelcount*sizeof(PathSegment),cudaMemcpyDeviceToDevice);
		checkCUDAError("generate camera ray");

	}else{
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter,traceDepth, dev_paths,Antialiasing);
	}
	
	
	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	setfinalbuffer<<<numBlocksPixels, blockSize1d>>>(num_paths,finalbuffer);
	//thrust::device_vector<int> dev_thrust_out(pixelcount);
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		SharcInit(sharcState);
		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		if(depth==0&&iter>1&&Cache){
			cudaMemcpy(dev_intersections,firstBounce,num_paths*sizeof(ShadeableIntersection),cudaMemcpyDeviceToDevice);
			depth++;
		}else{
			if(BVH){
				computeIntersectionsBVH << <numblocksPathSegmentTracing, blockSize1d >> > (
					iter
					, num_paths
					, dev_paths
					, dev_geoms
					, dev_tree
					, hst_scene->BVH.size()
					, dev_intersections
					);
			}else{
				computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
					iter
					, num_paths
					, dev_paths
					, dev_geoms
					, hst_scene->geoms.size()
					, dev_intersections
					);
			}
			
			
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
			depth++;
			
			if(sortMat){
				materialRemap << <numblocksPathSegmentTracing, blockSize1d >> >(
					num_paths,
					dev_paths,
					dev_intersections,
					dev_keys,
					dev_values
				);
				thrust::sort_by_key(thrust::device,dev_keys, dev_keys+num_paths, dev_values);
				kernReshuffle<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths,dev_values,dev_paths,dev_pathR,dev_intersections,dev_intersectionsR);
	
				PathSegment *tempdev_paths=dev_pathR;
				dev_pathR=dev_paths;
				dev_paths=tempdev_paths;
				ShadeableIntersection* tempdev_intersections=dev_intersectionsR;
				dev_intersectionsR=dev_intersections;
				dev_intersections=tempdev_intersections;
			}
		}

		if(depth==1&&iter==1&&Cache){
			cudaMemcpy(firstBounce,dev_intersections,num_paths*sizeof(ShadeableIntersection),cudaMemcpyDeviceToDevice);
			cudaMemcpy(firstBounceP,dev_paths,num_paths*sizeof(PathSegment),cudaMemcpyDeviceToDevice);
		}

		shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			depth,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_albedoImage,
			dev_normalImage,
			dev_materials,
			hst_scene->backColor,
			dev_env,
			dev_texture_objects,
			dev_geoms,
			dev_lights,
			dev_lights_area,
			hst_scene->Lights.size(),
			shading,
			sharcState
		);

		kernScatter << <numblocksPathSegmentTracing, blockSize1d >> > (
			num_paths,
			dev_paths,
			finalbuffer
		);
		
		PathSegment* new_end=thrust::remove_if(thrust::device,dev_paths,dev_paths+num_paths,is_zero());
		
		num_paths=new_end-dev_paths;

		if(depth>=traceDepth || num_paths==0){
			iterationComplete = true; // TODO: should be based off stream compaction results.
		}

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	
	}

		
	
	finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, finalbuffer);

	if(ENABLE_CACHE){
		// SHaRC Resolve and Compaction
		const int threadsPerBlock = 256;
		const int blocks = (sharcState.hashMapData.capacity + threadsPerBlock - 1) / threadsPerBlock;
		// SHaRC Resolve Kernel
		sharcState.gridParameters.cameraPositionPrev = sharcState.gridParameters.cameraPosition;
		sharcState.gridParameters.cameraPosition = glmToFloat3(cam.position);
		sharcResolveKernel << <blocks, threadsPerBlock >> >(
			dev_voxelDataBuffer, dev_voxelDataBufferPrev, dev_hashEntriesBuffer, 
			dev_copyOffsetBuffer, sharcState.gridParameters.cameraPosition, 
			sharcState.gridParameters.cameraPositionPrev, sharcState.gridParameters.sceneScale, 
			sharcState.hashMapData.capacity, 3, 32);
		cudaDeviceSynchronize();
		checkCUDAError("sharcResolveKernel");
		
		
		
		// SHaRC Compaction Kernel
		sharcCompactionKernel << <blocks, threadsPerBlock >> >(
			dev_hashEntriesBuffer, dev_copyOffsetBuffer, sharcState.hashMapData.capacity);
		cudaDeviceSynchronize();
		checkCUDAError("sharcCompactionKernel");
	}
	
	///////////////////////////////////////////////////////////////////////////

	// Assemble this iteration and apply it to the image
	if(DENOISE){
		int interval = 10;
		if (iter % interval == 0 || iter == hst_scene->state.iterations) {
			// copy image data to host memory
			// beauty buffer
			std::vector<glm::vec3> hst_image = std::vector<glm::vec3>(pixelcount);
			std::vector<glm::vec3> hst_image_post = std::vector<glm::vec3>(pixelcount);
			cudaMemcpy(hst_image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			cudaMemcpy(hst_image_post.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
			// albedo buffer
			std::vector<glm::vec3> hst_albedoImage = std::vector<glm::vec3>(pixelcount);
			cudaMemcpy(hst_albedoImage.data(), dev_albedoImage, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			// normal buffer
			std::vector<glm::vec3> hst_normalImage = std::vector<glm::vec3>(pixelcount);
			cudaMemcpy(hst_normalImage.data(), dev_normalImage, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			// denoise
			denoiseImage(&hst_image, &hst_albedoImage, &hst_normalImage, &hst_image_post, pixelcount, cam.resolution);
			// copy back to device memory
			cudaMemcpy(dev_image, hst_image_post.data(), pixelcount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
		}
    }

	tonemapKernel << <blocksPerGrid2d, blockSize2d >> > (dev_image, cam.resolution.x, cam.resolution.y, 2.2f);
	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
