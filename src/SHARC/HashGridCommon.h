//THE CODE BELOW IS TRANSLATED FROM HLSL/GLSL TO CUDA C++, ORIGINAL CODE FROM NVIDIA CORPORATION

#ifndef HASHGRIDCOMMON_H
#define HASHGRIDCOMMON_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <vector_functions.h>

#define HASH_GRID_POSITION_BIT_NUM          17
#define HASH_GRID_POSITION_BIT_MASK         ((1u << HASH_GRID_POSITION_BIT_NUM) - 1)
#define HASH_GRID_LEVEL_BIT_NUM             10
#define HASH_GRID_LEVEL_BIT_MASK            ((1u << HASH_GRID_LEVEL_BIT_NUM) - 1)
#define HASH_GRID_NORMAL_BIT_NUM            3
#define HASH_GRID_NORMAL_BIT_MASK           ((1u << HASH_GRID_NORMAL_BIT_NUM) - 1)
#define HASH_GRID_HASH_MAP_BUCKET_SIZE      32
#define HASH_GRID_INVALID_HASH_KEY          0
#define HASH_GRID_INVALID_CACHE_ENTRY       0xFFFFFFFF
#define HASH_GRID_USE_NORMALS               1
#define HASH_GRID_ALLOW_COMPACTION          (HASH_GRID_HASH_MAP_BUCKET_SIZE == 32)
#define HASH_GRID_LEVEL_BIAS                2
#define HASH_GRID_POSITION_OFFSET           make_float3(0.0f, 0.0f, 0.0f)
#define HASH_GRID_POSITION_BIAS             1e-4f
#define HASH_GRID_NORMAL_BIAS               1e-3f

#define CacheEntry uint
#define HashKey uint64_t

#ifndef uint
#define uint unsigned int
#endif

#ifndef uint64_t
#include <cstdint>
#endif

// Debug macros
#ifndef BUFFER_AT_OFFSET
#define BUFFER_AT_OFFSET(buffer, offset) buffer[offset]
#endif


#define RW_STRUCTURED_BUFFER(name, type) type* name

struct GridParameters {
    float3 cameraPosition;
    float3 cameraPositionPrev;
    float logarithmBase;
    float sceneScale;
};

__host__ __device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ int3 operator-(const int3& a, const int3& b) {
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ float3 operator*(const float3& a, const float& b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ float3 operator/(const float3& a, const float& b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

template <typename T>
__host__ __device__ inline T clamp(T value, T minValue, T maxValue) {
    return (value < minValue) ? minValue : (value > maxValue) ? maxValue : value;
}

__host__ __device__ inline int3 floor(const float3& vec) {
    return make_int3(floor(vec.x), floor(vec.y), floor(vec.z));
}

template <typename T>
__host__ __device__ inline T lerp(T v0, T v1, T t) {
    return v0 + t * (v1 - v0);
}

// Logarithm base function
__host__ __device__ inline float LogBase(float x, float base) {
    return logf(x) / logf(base);
}

// Get base slot for hash map compaction
__host__ __device__ inline uint GetBaseSlot(uint slot, uint capacity) {
#if HASH_GRID_ALLOW_COMPACTION
    return (slot / HASH_GRID_HASH_MAP_BUCKET_SIZE) * HASH_GRID_HASH_MAP_BUCKET_SIZE;
#else
    return min(slot, capacity - HASH_GRID_HASH_MAP_BUCKET_SIZE);
#endif
}

// Jenkins 32-bit hash function
__host__ __device__ inline uint HashJenkins32(uint a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// Combined hash function for 64-bit keys
__host__ __device__ inline uint Hash32(HashKey hashKey) {
    return HashJenkins32(uint((hashKey >> 0) & 0xffffffff))
         ^ HashJenkins32(uint((hashKey >> 32) & 0xffffffff));
}

// Get grid level based on distance to camera
__host__ __device__ inline uint GetGridLevel(float3 samplePosition, GridParameters gridParameters) {
    float3 diff = gridParameters.cameraPosition - samplePosition;
    float distance2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

    return uint(clamp(0.5f * LogBase(distance2, gridParameters.logarithmBase) + HASH_GRID_LEVEL_BIAS, 1.0f, float(HASH_GRID_LEVEL_BIT_MASK)));
}

// Calculate voxel size based on grid level
__host__ __device__ inline float GetVoxelSize(uint gridLevel, GridParameters gridParameters) {
    return powf(gridParameters.logarithmBase, gridLevel) / (gridParameters.sceneScale * powf(gridParameters.logarithmBase, HASH_GRID_LEVEL_BIAS));
}

// Calculate grid position in logarithmic space
__host__ __device__ inline int4 CalculateGridPositionLog(float3 samplePosition, GridParameters gridParameters) {
    samplePosition = samplePosition + make_float3(HASH_GRID_POSITION_BIAS, HASH_GRID_POSITION_BIAS, HASH_GRID_POSITION_BIAS);

    uint gridLevel = GetGridLevel(samplePosition, gridParameters);
    float voxelSize = GetVoxelSize(gridLevel, gridParameters);
    int3 gridPosition = make_int3(floorf(samplePosition.x / voxelSize),
                                  floorf(samplePosition.y / voxelSize),
                                  floorf(samplePosition.z / voxelSize));

    return make_int4(gridPosition.x, gridPosition.y, gridPosition.z, int(gridLevel));
}

// Compute spatial hash
__host__ __device__ inline HashKey ComputeSpatialHash(float3 samplePosition, float3 sampleNormal, GridParameters gridParameters) {
    int4 gridPosition = CalculateGridPositionLog(samplePosition, gridParameters);

    HashKey hashKey = ((uint64_t(gridPosition.x) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 0))
                    | ((uint64_t(gridPosition.y) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 1))
                    | ((uint64_t(gridPosition.z) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 2))
                    | ((uint64_t(gridPosition.w) & HASH_GRID_LEVEL_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 3));

#if HASH_GRID_USE_NORMALS
    uint normalBits =
        (sampleNormal.x + HASH_GRID_NORMAL_BIAS >= 0 ? 1 : 0) +
        (sampleNormal.y + HASH_GRID_NORMAL_BIAS >= 0 ? 2 : 0) +
        (sampleNormal.z + HASH_GRID_NORMAL_BIAS >= 0 ? 4 : 0);

    hashKey |= (uint64_t(normalBits) << (HASH_GRID_POSITION_BIT_NUM * 3 + HASH_GRID_LEVEL_BIT_NUM));
#endif

    return hashKey;
}

// Get position from hash key
__host__ __device__ inline float3 GetPositionFromHashKey(const HashKey hashKey, GridParameters gridParameters) {
    const int signBit = 1 << (HASH_GRID_POSITION_BIT_NUM - 1);
    const int signMask = ~((1 << HASH_GRID_POSITION_BIT_NUM) - 1);

    int3 gridPosition;
    gridPosition.x = int((hashKey >> (HASH_GRID_POSITION_BIT_NUM * 0)) & HASH_GRID_POSITION_BIT_MASK);
    gridPosition.y = int((hashKey >> (HASH_GRID_POSITION_BIT_NUM * 1)) & HASH_GRID_POSITION_BIT_MASK);
    gridPosition.z = int((hashKey >> (HASH_GRID_POSITION_BIT_NUM * 2)) & HASH_GRID_POSITION_BIT_MASK);

    // Fix negative coordinates
    gridPosition.x = (gridPosition.x & signBit) != 0 ? gridPosition.x | signMask : gridPosition.x;
    gridPosition.y = (gridPosition.y & signBit) != 0 ? gridPosition.y | signMask : gridPosition.y;
    gridPosition.z = (gridPosition.z & signBit) != 0 ? gridPosition.z | signMask : gridPosition.z;

    uint gridLevel = uint((hashKey >> HASH_GRID_POSITION_BIT_NUM * 3) & HASH_GRID_LEVEL_BIT_MASK);
    float voxelSize = GetVoxelSize(gridLevel, gridParameters);
    return make_float3((gridPosition.x + 0.5f) * voxelSize, (gridPosition.y + 0.5f) * voxelSize, (gridPosition.z + 0.5f) * voxelSize);
}

// Updated HashMapData structure for CUDA
struct HashMapData {
    uint capacity;

    RW_STRUCTURED_BUFFER(hashEntriesBuffer, uint64_t);

#if !HASH_GRID_ENABLE_64_BIT_ATOMICS
    RW_STRUCTURED_BUFFER(lockBuffer, uint);
#endif // !HASH_GRID_ENABLE_64_BIT_ATOMICS
};

// Atomic Compare Exchange
__device__ inline void HashMapAtomicCompareExchange(
    HashMapData& hashMapData, 
    uint dstOffset, 
    uint64_t compareValue, 
    uint64_t value, 
    uint64_t& originalValue) 
{
#if HASH_GRID_ENABLE_64_BIT_ATOMICS
    atomicCAS(&BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, dstOffset), compareValue, value);
    originalValue = BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, dstOffset);
#else
    const uint cLock = 0xAAAAAAAA;
    uint fuse = 0;
    const uint fuseLength = 8;
    bool busy = true;

    while (busy && fuse < fuseLength) {
        uint state = atomicExch(&BUFFER_AT_OFFSET(hashMapData.lockBuffer, dstOffset), cLock);
        busy = state != 0;

        if (state != cLock) {
            originalValue = BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, dstOffset);
            if (originalValue == compareValue) {
                BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, dstOffset) = value;
            }
            // Release the lock by setting it back to the previous state
            atomicExch(&BUFFER_AT_OFFSET(hashMapData.lockBuffer, dstOffset), state);
            // Exit the loop
            fuse = fuseLength;
        }
        ++fuse;
    }
#endif
}

// Insert function
__device__ inline bool HashMapInsert(
    HashMapData& hashMapData, 
    const uint64_t hashKey, 
    uint& cacheEntry) 
{
    uint hash = Hash32(hashKey);
    uint slot = hash % hashMapData.capacity;
    uint baseSlot = GetBaseSlot(slot, hashMapData.capacity);
    uint64_t prevHashKey = HASH_GRID_INVALID_HASH_KEY;

    for (uint bucketOffset = 0; bucketOffset < HASH_GRID_HASH_MAP_BUCKET_SIZE; ++bucketOffset) {
        HashMapAtomicCompareExchange(hashMapData, baseSlot + bucketOffset, HASH_GRID_INVALID_HASH_KEY, hashKey, prevHashKey);

        if (prevHashKey == HASH_GRID_INVALID_HASH_KEY || prevHashKey == hashKey) {
            cacheEntry = baseSlot + bucketOffset;
            return true;
        }
    }

    cacheEntry = HASH_GRID_INVALID_CACHE_ENTRY;
    return false;
}

// Find function
__host__ __device__ inline bool HashMapFind(
    const HashMapData& hashMapData, 
    const uint64_t hashKey, 
    uint& cacheEntry) 
{
    uint hash = Hash32(hashKey);
    uint slot = hash % hashMapData.capacity;
    uint baseSlot = GetBaseSlot(slot, hashMapData.capacity);

    for (uint bucketOffset = 0; bucketOffset < HASH_GRID_HASH_MAP_BUCKET_SIZE; ++bucketOffset) {
        uint64_t storedHashKey = BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, baseSlot + bucketOffset);

        if (storedHashKey == hashKey) {
            cacheEntry = baseSlot + bucketOffset;
            return true;
        }
#if HASH_GRID_ALLOW_COMPACTION
        else if (storedHashKey == HASH_GRID_INVALID_HASH_KEY) {
            return false;
        }
#endif
    }

    return false;
}

// Insert Entry
__device__ inline uint HashMapInsertEntry(
    HashMapData& hashMapData, 
    const float3 samplePosition, 
    const float3 sampleNormal, 
    GridParameters& gridParameters) 
{
    uint cacheEntry = HASH_GRID_INVALID_CACHE_ENTRY;
    uint64_t hashKey = ComputeSpatialHash(samplePosition, sampleNormal, gridParameters);
    HashMapInsert(hashMapData, hashKey, cacheEntry);
    return cacheEntry;
}

__host__ __device__ inline CacheEntry HashMapFindEntry(
    const HashMapData& hashMapData,
    const float3& samplePosition,
    const float3& sampleNormal,
    const GridParameters& gridParameters
) {
    CacheEntry cacheEntry = HASH_GRID_INVALID_CACHE_ENTRY;
    const HashKey hashKey = ComputeSpatialHash(samplePosition, sampleNormal, gridParameters);
    bool successful = HashMapFind(hashMapData, hashKey, cacheEntry);

    return cacheEntry;
}

// Debug functions
__device__ inline float3 GetColorFromHash32(uint hash) {
    return make_float3(
        ((hash >> 0) & 0x3ff) / 1023.0f,
        ((hash >> 11) & 0x7ff) / 2047.0f,
        ((hash >> 22) & 0x7ff) / 2047.0f
    );
}

__device__ inline float3 HashGridDebugColoredHash(
    float3 samplePosition, 
    GridParameters gridParameters) 
{
    uint64_t hashKey = ComputeSpatialHash(samplePosition, make_float3(0, 0, 0), gridParameters);
    uint gridLevel = GetGridLevel(samplePosition, gridParameters);
    return GetColorFromHash32(Hash32(hashKey)) * GetColorFromHash32(HashJenkins32(gridLevel));
}

__device__ inline float3 HashGridDebugOccupancy(
    uint2 pixelPosition, 
    uint2 screenSize, 
    HashMapData& hashMapData) 
{
    const uint elementSize = 7;
    const uint borderSize = 1;
    const uint blockSize = elementSize + borderSize;

    uint rowNum = screenSize.y / blockSize;
    uint rowIndex = pixelPosition.y / blockSize;
    uint columnIndex = pixelPosition.x / blockSize;
    uint elementIndex = (columnIndex / HASH_GRID_HASH_MAP_BUCKET_SIZE) * (rowNum * HASH_GRID_HASH_MAP_BUCKET_SIZE) + rowIndex * HASH_GRID_HASH_MAP_BUCKET_SIZE + (columnIndex % HASH_GRID_HASH_MAP_BUCKET_SIZE);

    if (elementIndex < hashMapData.capacity &&
        ((pixelPosition.x % blockSize) < elementSize && 
         (pixelPosition.y % blockSize) < elementSize)) 
    {
        uint64_t storedHashKey = BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, elementIndex);
        if (storedHashKey != HASH_GRID_INVALID_HASH_KEY)
            return make_float3(0.0f, 1.0f, 0.0f);
    }

    return make_float3(0.0f, 0.0f, 0.0f);
}

#endif // HASHGRIDCOMMON_H
