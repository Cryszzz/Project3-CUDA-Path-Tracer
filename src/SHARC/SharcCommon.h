//THE CODE BELOW IS TRANSLATED FROM HLSL/GLSL TO CUDA C++, ORIGINAL CODE FROM NVIDIA CORPORATION

/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

// Version definitions
#define SHARC_VERSION_MAJOR                 1
#define SHARC_VERSION_MINOR                 3
#define SHARC_VERSION_BUILD                 1
#define SHARC_VERSION_REVISION              0

// Define SHARC_UPDATE and SHARC_QUERY based on their states
#if (SHARC_UPDATE || SHARC_QUERY)
    #if SHARC_UPDATE
        #define SHARC_QUERY 0
    #else
        #define SHARC_UPDATE 0
    #endif
#else
    #define SHARC_QUERY 0
    #define SHARC_UPDATE 0
#endif

// Constants for SHARC
#define SHARC_SAMPLE_NUM_MULTIPLIER             16
#define SHARC_SAMPLE_NUM_THRESHOLD              0
#define SHARC_SEPARATE_EMISSIVE                 0
#define SHARC_PROPOGATION_DEPTH                 4
#define SHARC_ENABLE_CACHE_RESAMPLING           (SHARC_UPDATE && (SHARC_PROPOGATION_DEPTH > 1))
#define SHARC_RESAMPLING_DEPTH_MIN              1
#define SHARC_RADIANCE_SCALE                    1e3f
#define SHARC_ACCUMULATED_FRAME_NUM_MIN         1
#define SHARC_ACCUMULATED_FRAME_NUM_MAX         64
#define SHARC_STALE_FRAME_NUM_MIN               32

// Bit mask and offset configurations
#define SHARC_SAMPLE_NUM_BIT_NUM                18
#define SHARC_SAMPLE_NUM_BIT_OFFSET             0
#define SHARC_SAMPLE_NUM_BIT_MASK               ((1u << SHARC_SAMPLE_NUM_BIT_NUM) - 1)

#define SHARC_ACCUMULATED_FRAME_NUM_BIT_NUM     6
#define SHARC_ACCUMULATED_FRAME_NUM_BIT_OFFSET  (SHARC_SAMPLE_NUM_BIT_NUM)
#define SHARC_ACCUMULATED_FRAME_NUM_BIT_MASK    ((1u << SHARC_ACCUMULATED_FRAME_NUM_BIT_NUM) - 1)

#define SHARC_STALE_FRAME_NUM_BIT_NUM           8
#define SHARC_STALE_FRAME_NUM_BIT_OFFSET        (SHARC_SAMPLE_NUM_BIT_NUM + SHARC_ACCUMULATED_FRAME_NUM_BIT_NUM)
#define SHARC_STALE_FRAME_NUM_BIT_MASK          ((1u << SHARC_STALE_FRAME_NUM_BIT_NUM) - 1)

#define SHARC_GRID_LOGARITHM_BASE               2.0f
#define SHARC_ENABLE_COMPACTION                 HASH_GRID_ALLOW_COMPACTION
#define SHARC_BLEND_ADJACENT_LEVELS             1
#define SHARC_DEFERRED_HASH_COMPACTION          (SHARC_ENABLE_COMPACTION && SHARC_BLEND_ADJACENT_LEVELS)
#define SHARC_NORMALIZED_SAMPLE_NUM             (1u << (SHARC_SAMPLE_NUM_BIT_NUM - 1))

// Debugging thresholds
#define SHARC_DEBUG_BITS_OCCUPANCY_THRESHOLD_LOW        0.125f
#define SHARC_DEBUG_BITS_OCCUPANCY_THRESHOLD_MEDIUM     0.5f

// Define the RW_STRUCTURED_BUFFER macro
#ifndef RW_STRUCTURED_BUFFER
    #define RW_STRUCTURED_BUFFER(name, type) type* name
#endif

// Includes
#include "HashGridCommon.h"
#include <device_functions.h>

// Structures
struct SharcVoxelData {
    uint3 accumulatedRadiance;
    uint accumulatedSampleNum;
    uint accumulatedFrameNum;
    uint staleFrameNum;
};

struct SharcHitData {
    float3 positionWorld;
    float3 normalWorld;
    #if SHARC_SEPARATE_EMISSIVE
    float3 emissive;
    #endif
};

// Utility functions
__host__ __device__ inline uint SharcGetSampleNum(uint packedData) {
    return (packedData >> SHARC_SAMPLE_NUM_BIT_OFFSET) & SHARC_SAMPLE_NUM_BIT_MASK;
}

__host__ __device__ inline uint SharcGetStaleFrameNum(uint packedData) {
    return (packedData >> SHARC_STALE_FRAME_NUM_BIT_OFFSET) & SHARC_STALE_FRAME_NUM_BIT_MASK;
}

__host__ __device__ inline uint SharcGetAccumulatedFrameNum(uint packedData) {
    return (packedData >> SHARC_ACCUMULATED_FRAME_NUM_BIT_OFFSET) & SHARC_ACCUMULATED_FRAME_NUM_BIT_MASK;
}

__host__ __device__ inline float3 SharcResolveAccumulatedRadiance(uint3 accumulatedRadiance, uint accumulatedSampleNum) {
    return make_float3(accumulatedRadiance.x, accumulatedRadiance.y, accumulatedRadiance.z) / (accumulatedSampleNum * SHARC_RADIANCE_SCALE);
}

__host__ __device__ inline SharcVoxelData SharcUnpackVoxelData(uint4 voxelDataPacked) {
    SharcVoxelData voxelData;
    voxelData.accumulatedRadiance = make_uint3(voxelDataPacked.x, voxelDataPacked.y, voxelDataPacked.z);
    voxelData.accumulatedSampleNum = SharcGetSampleNum(voxelDataPacked.w);
    voxelData.staleFrameNum = SharcGetStaleFrameNum(voxelDataPacked.w);
    voxelData.accumulatedFrameNum = SharcGetAccumulatedFrameNum(voxelDataPacked.w);
    return voxelData;
}

__host__ __device__ inline SharcVoxelData SharcGetVoxelData(uint4* voxelDataBuffer, CacheEntry cacheEntry) {
    SharcVoxelData voxelData;
    voxelData.accumulatedRadiance = make_uint3(0, 0, 0);
    voxelData.accumulatedSampleNum = 0;
    voxelData.accumulatedFrameNum = 0;
    voxelData.staleFrameNum = 0;

    // Check for invalid cache entry
    if (cacheEntry == HASH_GRID_INVALID_CACHE_ENTRY) {
        return voxelData;
    }

    // Fetch packed data from buffer
    uint4 voxelDataPacked = voxelDataBuffer[cacheEntry];

    // Unpack and return the voxel data
    voxelData.accumulatedRadiance = make_uint3(voxelDataPacked.x, voxelDataPacked.y, voxelDataPacked.z);
    voxelData.accumulatedSampleNum = (voxelDataPacked.w >> SHARC_SAMPLE_NUM_BIT_OFFSET) & SHARC_SAMPLE_NUM_BIT_MASK;
    voxelData.accumulatedFrameNum = (voxelDataPacked.w >> SHARC_ACCUMULATED_FRAME_NUM_BIT_OFFSET) & SHARC_ACCUMULATED_FRAME_NUM_BIT_MASK;
    voxelData.staleFrameNum = (voxelDataPacked.w >> SHARC_STALE_FRAME_NUM_BIT_OFFSET) & SHARC_STALE_FRAME_NUM_BIT_MASK;

    return voxelData;
}

// Additional utility functions
__device__ inline void SharcAddVoxelData(
    uint4* voxelDataBuffer, CacheEntry cacheEntry, float3 value, uint sampleData) {
    if (cacheEntry == HASH_GRID_INVALID_CACHE_ENTRY)
        return;

    uint3 scaledRadiance = make_uint3(
        static_cast<unsigned int>(value.x * SHARC_RADIANCE_SCALE),
        static_cast<unsigned int>(value.y * SHARC_RADIANCE_SCALE),
        static_cast<unsigned int>(value.z * SHARC_RADIANCE_SCALE)
    );

    atomicAdd(&voxelDataBuffer[cacheEntry].x, scaledRadiance.x);
    atomicAdd(&voxelDataBuffer[cacheEntry].y, scaledRadiance.y);
    atomicAdd(&voxelDataBuffer[cacheEntry].z, scaledRadiance.z);
    atomicAdd(&voxelDataBuffer[cacheEntry].w, sampleData);
}

struct SharcState {
    GridParameters gridParameters;
    HashMapData hashMapData;

    #if SHARC_UPDATE
    CacheEntry cacheEntry[SHARC_PROPOGATION_DEPTH];
    float3 sampleWeight[SHARC_PROPOGATION_DEPTH];
    uint pathLength;
    #endif

    RW_STRUCTURED_BUFFER(voxelDataBuffer, uint4);

    #if SHARC_ENABLE_CACHE_RESAMPLING
    RW_STRUCTURED_BUFFER(voxelDataBufferPrev, uint4);
    #endif
};

// Initialize SHARC state
__host__ __device__ inline void SharcInit(SharcState& sharcState) {
    #if SHARC_UPDATE
    sharcState.pathLength = 0;
    #endif
}

__device__ void SharcUpdateMiss(SharcState& sharcState, const float3& radiance) {
#if SHARC_UPDATE
    float3 currentRadiance = radiance;
    for (int i = 0; i < sharcState.pathLength; ++i) {
        currentRadiance = currentRadiance* sharcState.sampleWeight[i];
        SharcAddVoxelData(sharcState.voxelDataBuffer, sharcState.cacheEntry[i], currentRadiance, 0);
    }
#endif // SHARC_UPDATE
}

__device__ bool SharcUpdateHit(SharcState& sharcState, const SharcHitData& sharcHitData, float3 lighting, float random) {
    bool continueTracing = true;
#if SHARC_UPDATE
    CacheEntry cacheEntry = HashMapInsertEntry(sharcState.hashMapData, sharcHitData.positionWorld, sharcHitData.normalWorld, sharcState.gridParameters);

    float3 sharcRadiance = lighting;

#if SHARC_ENABLE_CACHE_RESAMPLING
    uint resamplingDepth = uint(round(lerp((float)SHARC_RESAMPLING_DEPTH_MIN, (float)SHARC_PROPOGATION_DEPTH - 1.0f, random)));
    if (resamplingDepth <= sharcState.pathLength) {
        SharcVoxelData voxelData = SharcGetVoxelData(sharcState.voxelDataBufferPrev, cacheEntry);
        if (voxelData.accumulatedSampleNum > SHARC_SAMPLE_NUM_THRESHOLD) {
            sharcRadiance = SharcResolveAccumulatedRadiance(voxelData.accumulatedRadiance, voxelData.accumulatedSampleNum);
            continueTracing = false;
        }
    }
#endif // SHARC_ENABLE_CACHE_RESAMPLING

    if (continueTracing) {
        SharcAddVoxelData(sharcState.voxelDataBuffer, cacheEntry, lighting, 1);
    }

#if SHARC_SEPARATE_EMISSIVE
    sharcRadiance += sharcHitData.emissive;
#endif // SHARC_SEPARATE_EMISSIVE

    for (uint i = 0; i < sharcState.pathLength; ++i) {
        sharcRadiance = sharcRadiance*sharcState.sampleWeight[i];
        SharcAddVoxelData(sharcState.voxelDataBuffer, sharcState.cacheEntry[i], sharcRadiance, 0);
    }

    for (uint i = sharcState.pathLength; i > 0; --i) {
        sharcState.cacheEntry[i] = sharcState.cacheEntry[i - 1];
        sharcState.sampleWeight[i] = sharcState.sampleWeight[i - 1];
    }

    sharcState.cacheEntry[0] = cacheEntry;
    sharcState.pathLength = min(sharcState.pathLength + 1, (unsigned int)SHARC_PROPOGATION_DEPTH - 1);
#endif // SHARC_UPDATE
    return continueTracing;
}

__device__ void SharcSetThroughput(SharcState& sharcState, const float3& throughput) {
#if SHARC_UPDATE
    sharcState.sampleWeight[0] = throughput;
#endif // SHARC_UPDATE
}

__device__ bool SharcGetCachedRadiance(const SharcState& sharcState, const SharcHitData& sharcHitData, float3& radiance, bool debug) {
    if (debug) radiance = make_float3(0.0f, 0.0f, 0.0f);
    const uint sampleThreshold = debug ? 0 : SHARC_SAMPLE_NUM_THRESHOLD;

    CacheEntry cacheEntry = HashMapFindEntry(sharcState.hashMapData, sharcHitData.positionWorld, sharcHitData.normalWorld, sharcState.gridParameters);
    if (cacheEntry == HASH_GRID_INVALID_CACHE_ENTRY) {
        return false;
    }

    SharcVoxelData voxelData = SharcGetVoxelData(sharcState.voxelDataBuffer, cacheEntry);
    if (voxelData.accumulatedSampleNum > sampleThreshold) {
        radiance = SharcResolveAccumulatedRadiance(voxelData.accumulatedRadiance, voxelData.accumulatedSampleNum);

#if SHARC_SEPARATE_EMISSIVE
        radiance += sharcHitData.emissive;
#endif // SHARC_SEPARATE_EMISSIVE

        return true;
    }
    return false;
}

__device__ void SharcCopyHashEntry(uint entryIndex, HashMapData hashMapData, uint* copyOffsetBuffer) {
#if SHARC_DEFERRED_HASH_COMPACTION
    if (entryIndex >= hashMapData.capacity) return;

    uint copyOffset = copyOffsetBuffer[entryIndex];
    if (copyOffset == 0) return;

    if (copyOffset == HASH_GRID_INVALID_CACHE_ENTRY) {
        hashMapData.hashEntriesBuffer[entryIndex] = HASH_GRID_INVALID_HASH_KEY;
    } else {
        HashKey hashKey = hashMapData.hashEntriesBuffer[entryIndex];
        hashMapData.hashEntriesBuffer[entryIndex] = HASH_GRID_INVALID_HASH_KEY;
        hashMapData.hashEntriesBuffer[copyOffset] = hashKey;
    }
    copyOffsetBuffer[entryIndex] = 0;
#endif // SHARC_DEFERRED_HASH_COMPACTION
}

__device__ int SharcGetGridDistance2(const int3& position) {
    return position.x * position.x + position.y * position.y + position.z * position.z;
}

__device__ HashKey SharcGetAdjacentLevelHashKey(HashKey hashKey, const GridParameters& gridParameters) {
    const int signBit = 1 << (HASH_GRID_POSITION_BIT_NUM - 1);
    const int signMask = ~((1 << HASH_GRID_POSITION_BIT_NUM) - 1);

    int3 gridPosition;
    gridPosition.x = int((hashKey >> (HASH_GRID_POSITION_BIT_NUM * 0)) & HASH_GRID_POSITION_BIT_MASK);
    gridPosition.y = int((hashKey >> (HASH_GRID_POSITION_BIT_NUM * 1)) & HASH_GRID_POSITION_BIT_MASK);
    gridPosition.z = int((hashKey >> (HASH_GRID_POSITION_BIT_NUM * 2)) & HASH_GRID_POSITION_BIT_MASK);

    gridPosition.x = (gridPosition.x & signBit) ? gridPosition.x | signMask : gridPosition.x;
    gridPosition.y = (gridPosition.y & signBit) ? gridPosition.y | signMask : gridPosition.y;
    gridPosition.z = (gridPosition.z & signBit) ? gridPosition.z | signMask : gridPosition.z;

    int level = int((hashKey >> (HASH_GRID_POSITION_BIT_NUM * 3)) & HASH_GRID_LEVEL_BIT_MASK);

    float voxelSize = GetVoxelSize(level, gridParameters);
    int3 cameraGridPosition = floor((gridParameters.cameraPosition + HASH_GRID_POSITION_OFFSET) / voxelSize);
    int cameraDistance = SharcGetGridDistance2(cameraGridPosition - gridPosition);

    int3 cameraGridPositionPrev = floor((gridParameters.cameraPositionPrev + HASH_GRID_POSITION_OFFSET) / voxelSize);
    int cameraDistancePrev = SharcGetGridDistance2(cameraGridPositionPrev - gridPosition);

    if (cameraDistance < cameraDistancePrev) {
        gridPosition = floor(make_float3(gridPosition.x, gridPosition.y, gridPosition.z) / gridParameters.logarithmBase);
        level = min(level + 1, int(HASH_GRID_LEVEL_BIT_MASK));
    } else {
        gridPosition = floor(make_float3(gridPosition.x, gridPosition.y, gridPosition.z) * gridParameters.logarithmBase);
        level = max(level - 1, 1);
    }

    HashKey modifiedHashKey = ((uint64_t(gridPosition.x) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 0))
                            | ((uint64_t(gridPosition.y) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 1))
                            | ((uint64_t(gridPosition.z) & HASH_GRID_POSITION_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 2))
                            | ((uint64_t(level) & HASH_GRID_LEVEL_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 3));

#if HASH_GRID_USE_NORMALS
    modifiedHashKey |= hashKey & (uint64_t(HASH_GRID_NORMAL_BIT_MASK) << (HASH_GRID_POSITION_BIT_NUM * 3 + HASH_GRID_LEVEL_BIT_NUM));
#endif // HASH_GRID_USE_NORMALS

    return modifiedHashKey;
}



// Resolve accumulated radiance
__host__ __device__ inline bool SharcResolveEntry(
    uint entryIndex, GridParameters gridParameters, HashMapData hashMapData,
    uint4* voxelDataBuffer, uint4* voxelDataBufferPrev, uint accumulationFrameNum, uint staleFrameNumMax) {
    if (entryIndex >= hashMapData.capacity)
        return false;

    HashKey hashKey = hashMapData.hashEntriesBuffer[entryIndex];
    if (hashKey == HASH_GRID_INVALID_HASH_KEY)
        return false;

    uint4 voxelDataPackedPrev = voxelDataBufferPrev[entryIndex];
    uint4 voxelDataPacked = voxelDataBuffer[entryIndex];

    uint sampleNum = SharcGetSampleNum(voxelDataPacked.w);
    uint sampleNumPrev = SharcGetSampleNum(voxelDataPackedPrev.w);
    uint accumulatedFrameNum = SharcGetAccumulatedFrameNum(voxelDataPackedPrev.w);
    uint staleFrameNum = SharcGetStaleFrameNum(voxelDataPackedPrev.w);

    uint3 accumulatedRadiance = make_uint3(
        voxelDataPacked.x * SHARC_SAMPLE_NUM_MULTIPLIER + voxelDataPackedPrev.x,
        voxelDataPacked.y * SHARC_SAMPLE_NUM_MULTIPLIER + voxelDataPackedPrev.y,
        voxelDataPacked.z * SHARC_SAMPLE_NUM_MULTIPLIER + voxelDataPackedPrev.z
    );

    uint accumulatedSampleNum = sampleNum * SHARC_SAMPLE_NUM_MULTIPLIER + sampleNumPrev;

    // Clamp to avoid overflow
    if (accumulatedSampleNum > SHARC_NORMALIZED_SAMPLE_NUM) {
        accumulatedSampleNum >>= 1;
        accumulatedRadiance.x >>= 1;
        accumulatedRadiance.y >>= 1;
        accumulatedRadiance.z >>= 1;
    }

    // Update accumulation frame and stale frame numbers
    accumulatedFrameNum = clamp(
        accumulatedFrameNum,
        static_cast<uint>(SHARC_ACCUMULATED_FRAME_NUM_MIN),
        static_cast<uint>(SHARC_ACCUMULATED_FRAME_NUM_MAX)
    );
    ++accumulatedFrameNum;
    staleFrameNum = (sampleNum != 0) ? 0 : staleFrameNum + 1;

    // Pack data
    uint4 packedData = make_uint4(accumulatedRadiance.x, accumulatedRadiance.y, accumulatedRadiance.z,
                                  (min(accumulatedSampleNum, SHARC_SAMPLE_NUM_BIT_MASK) |
                                   (min(accumulatedFrameNum, SHARC_ACCUMULATED_FRAME_NUM_BIT_MASK) << SHARC_ACCUMULATED_FRAME_NUM_BIT_OFFSET) |
                                   (min(staleFrameNum, SHARC_STALE_FRAME_NUM_BIT_MASK) << SHARC_STALE_FRAME_NUM_BIT_OFFSET)));

    // Update the buffer with valid or invalid data
    if (staleFrameNum >= max(staleFrameNumMax, (unsigned int)SHARC_STALE_FRAME_NUM_MIN)) {
        voxelDataBuffer[entryIndex] = make_uint4(0, 0, 0, 0);
        return false;
    } else {
        voxelDataBuffer[entryIndex] = packedData;
        return true;
    }
}

// Debugging utility functions
__host__ __device__ inline float3 SharcDebugGetBitsOccupancyColor(float occupancy) {
    if (occupancy < SHARC_DEBUG_BITS_OCCUPANCY_THRESHOLD_LOW) {
        return make_float3(0.0f, 1.0f, 0.0f) * (occupancy + SHARC_DEBUG_BITS_OCCUPANCY_THRESHOLD_LOW);
    } else if (occupancy < SHARC_DEBUG_BITS_OCCUPANCY_THRESHOLD_MEDIUM) {
        return make_float3(1.0f, 1.0f, 0.0f) * (occupancy + SHARC_DEBUG_BITS_OCCUPANCY_THRESHOLD_MEDIUM);
    } else {
        return make_float3(1.0f, 0.0f, 0.0f) * occupancy;
    }
}

// Debug visualization for sample numbers
__host__ __device__ inline float3 SharcDebugBitsOccupancySampleNum(
    const SharcState& sharcState, const SharcHitData& sharcHitData) {
    CacheEntry cacheEntry = HashMapFindEntry(
        sharcState.hashMapData, sharcHitData.positionWorld, sharcHitData.normalWorld, sharcState.gridParameters);
    SharcVoxelData voxelData = SharcGetVoxelData(sharcState.voxelDataBuffer, cacheEntry);

    float occupancy = static_cast<float>(voxelData.accumulatedSampleNum) / SHARC_SAMPLE_NUM_BIT_MASK;
    return SharcDebugGetBitsOccupancyColor(occupancy);
}

// Debug visualization for radiance
__host__ __device__ inline float3 SharcDebugBitsOccupancyRadiance(
    const SharcState& sharcState, const SharcHitData& sharcHitData) {
    CacheEntry cacheEntry = HashMapFindEntry(
        sharcState.hashMapData, sharcHitData.positionWorld, sharcHitData.normalWorld, sharcState.gridParameters);
    SharcVoxelData voxelData = SharcGetVoxelData(sharcState.voxelDataBuffer, cacheEntry);

    float maxRadiance = fmaxf(voxelData.accumulatedRadiance.x,
        fmaxf(voxelData.accumulatedRadiance.y, voxelData.accumulatedRadiance.z));
    float occupancy = maxRadiance / 0xffffffff;
    return SharcDebugGetBitsOccupancyColor(occupancy);
}
