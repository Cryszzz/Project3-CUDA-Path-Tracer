//THE CODE BELOW IS TRANSLATED FROM HLSL/GLSL TO CUDA C++, ORIGINAL CODE FROM NVIDIA :https://github.com/NVIDIAGameWorks/SHARC

// Version definitions
#define SHARC_VERSION_MAJOR                 1
#define SHARC_VERSION_MINOR                 3
#define SHARC_VERSION_BUILD                 1
#define SHARC_VERSION_REVISION              0

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

#ifdef __CUDACC__
    #define InterlockedAdd(address, value) atomicAdd(address, value)
#endif

#ifdef __CUDACC__
    #define InterlockedCompareExchange(address, compare, value) atomicCAS(address, compare, value)
#endif

// Warp-synchronous operations for CUDA
#define FULL_MASK 0xFFFFFFFF

// Count active bits in the warp ballot
#define WaveActiveCountBits(value) __popc(__ballot_sync(FULL_MASK, value))

// Perform a warp-wide ballot, returning a bitmask
#define WaveActiveBallot(value) __ballot_sync(FULL_MASK, value)

// Compute the prefix count (exclusive) of active bits up to the current thread in the warp
#define WavePrefixCountBits(value) __popc(__ballot_sync(FULL_MASK, value) & ((1U << threadIdx.x) - 1))


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

    uint4 voxelDataPacked = BUFFER_AT_OFFSET(voxelDataBuffer, cacheEntry);

    return SharcUnpackVoxelData(voxelDataPacked);
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

    if (scaledRadiance.x != 0) atomicAdd(&voxelDataBuffer[cacheEntry].x, scaledRadiance.x);
    if (scaledRadiance.y != 0) atomicAdd(&voxelDataBuffer[cacheEntry].y, scaledRadiance.y);
    if (scaledRadiance.z != 0) atomicAdd(&voxelDataBuffer[cacheEntry].z, scaledRadiance.z);
    if (sampleData != 0) atomicAdd(&voxelDataBuffer[cacheEntry].w, sampleData);
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
    sharcRadiance = sharcRadiance+sharcHitData.emissive;
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
    sharcState.pathLength = min(++sharcState.pathLength, (unsigned int)SHARC_PROPOGATION_DEPTH - 1);
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
        radiance = radiance+ sharcHitData.emissive;
#endif // SHARC_SEPARATE_EMISSIVE

        return true;
    }
    return false;
}

__device__ void SharcCopyHashEntry(uint entryIndex, HashMapData hashMapData, RW_STRUCTURED_BUFFER(copyOffsetBuffer, uint)) {
#if SHARC_DEFERRED_HASH_COMPACTION
    if (entryIndex >= hashMapData.capacity)
        return;

    uint copyOffset = BUFFER_AT_OFFSET(copyOffsetBuffer, entryIndex);
    if (copyOffset == 0)
        return;

    if (copyOffset == HASH_GRID_INVALID_CACHE_ENTRY)
    {
        BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, entryIndex) = HASH_GRID_INVALID_HASH_KEY;
    }
    else if (copyOffset != 0)
    {
        HashKey hashKey = BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, entryIndex);
        BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, entryIndex) = HASH_GRID_INVALID_HASH_KEY;
        BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, copyOffset) = hashKey;
    }

    BUFFER_AT_OFFSET(copyOffsetBuffer, entryIndex) = 0;
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



__device__ void SharcResolveEntry(uint entryIndex, GridParameters gridParameters, HashMapData hashMapData, RW_STRUCTURED_BUFFER(copyOffsetBuffer, uint),
    RW_STRUCTURED_BUFFER(voxelDataBuffer, uint4), RW_STRUCTURED_BUFFER(voxelDataBufferPrev, uint4), uint accumulationFrameNum, uint staleFrameNumMax)
{
    if (entryIndex >= hashMapData.capacity)
        return;

    HashKey hashKey = BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, entryIndex);
    if (hashKey == HASH_GRID_INVALID_HASH_KEY)
        return;

    uint4 voxelDataPackedPrev = BUFFER_AT_OFFSET(voxelDataBufferPrev, entryIndex);
    uint4 voxelDataPacked = BUFFER_AT_OFFSET(voxelDataBuffer, entryIndex);

    uint sampleNum = SharcGetSampleNum(voxelDataPacked.w);
    uint sampleNumPrev = SharcGetSampleNum(voxelDataPackedPrev.w);
    uint accumulatedFrameNum = SharcGetAccumulatedFrameNum(voxelDataPackedPrev.w);
    uint staleFrameNum = SharcGetStaleFrameNum(voxelDataPackedPrev.w);

    uint3 accumulatedRadiance = make_uint3(voxelDataPacked.x* SHARC_SAMPLE_NUM_MULTIPLIER+ voxelDataPackedPrev.x, voxelDataPacked.y* SHARC_SAMPLE_NUM_MULTIPLIER+voxelDataPackedPrev.y, voxelDataPacked.z* SHARC_SAMPLE_NUM_MULTIPLIER+voxelDataPackedPrev.z);
    uint accumulatedSampleNum = SharcGetSampleNum(voxelDataPacked.w) * SHARC_SAMPLE_NUM_MULTIPLIER + SharcGetSampleNum(voxelDataPackedPrev.w);

#if SHARC_BLEND_ADJACENT_LEVELS
    // Reproject sample from adjacent level
    float3 cameraOffset = make_float3(gridParameters.cameraPosition.x - gridParameters.cameraPositionPrev.x,
                                      gridParameters.cameraPosition.y - gridParameters.cameraPositionPrev.y,
                                      gridParameters.cameraPosition.z - gridParameters.cameraPositionPrev.z);
    if ((dot(cameraOffset, cameraOffset) != 0) && (accumulatedFrameNum < accumulationFrameNum))
    {
        HashKey adjacentLevelHashKey = SharcGetAdjacentLevelHashKey(hashKey, gridParameters);

        CacheEntry cacheEntry = HASH_GRID_INVALID_CACHE_ENTRY;
        if (HashMapFind(hashMapData, adjacentLevelHashKey, cacheEntry))
        {
            uint4 adjacentPackedDataPrev = BUFFER_AT_OFFSET(voxelDataBufferPrev, cacheEntry);
            uint adjacentSampleNum = SharcGetSampleNum(adjacentPackedDataPrev.w);
            if (adjacentSampleNum > SHARC_SAMPLE_NUM_THRESHOLD)
            {
                float blendWeight = adjacentSampleNum / float(adjacentSampleNum + accumulatedSampleNum);
                accumulatedRadiance = lerp(make_float3(accumulatedRadiance.x, accumulatedRadiance.y, accumulatedRadiance.z), make_float3(adjacentPackedDataPrev.x, adjacentPackedDataPrev.y, adjacentPackedDataPrev.z), blendWeight);
                accumulatedSampleNum = uint(lerp(float(accumulatedSampleNum), float(adjacentSampleNum), blendWeight));
            }
        }
    }
#endif // SHARC_BLEND_ADJACENT_LEVELS

    // Clamp internal sample count to help with potential overflow
    if (accumulatedSampleNum > SHARC_NORMALIZED_SAMPLE_NUM)
    {
        accumulatedSampleNum = accumulatedSampleNum>>1;
        accumulatedRadiance.x = accumulatedRadiance.x >> 1;
        accumulatedRadiance.y = accumulatedRadiance.y >> 1;
        accumulatedRadiance.z = accumulatedRadiance.z >> 1;
    }

    accumulationFrameNum = max(min(accumulationFrameNum, (unsigned int)SHARC_ACCUMULATED_FRAME_NUM_MAX), (unsigned int)SHARC_ACCUMULATED_FRAME_NUM_MIN);
    if (accumulatedFrameNum > accumulationFrameNum)
    {
        float normalizedAccumulatedSampleNum = round(accumulatedSampleNum * float(accumulationFrameNum) / accumulatedFrameNum);
        float normalizationScale = normalizedAccumulatedSampleNum / accumulatedSampleNum;

        accumulatedSampleNum = uint(normalizedAccumulatedSampleNum);
        accumulatedRadiance = make_uint3(accumulatedRadiance.x * normalizationScale, accumulatedRadiance.y * normalizationScale, accumulatedRadiance.z * normalizationScale);
        accumulatedFrameNum = uint(accumulatedFrameNum * normalizationScale);
    }

    ++accumulatedFrameNum;
    staleFrameNum = (sampleNum != 0) ? 0 : staleFrameNum + 1;

    uint4 packedData;
    packedData.x = accumulatedRadiance.x;
    packedData.y = accumulatedRadiance.y;
    packedData.z = accumulatedRadiance.z;

    packedData.w = min(accumulatedSampleNum, SHARC_SAMPLE_NUM_BIT_MASK);
    packedData.w |= (min(accumulatedFrameNum, SHARC_ACCUMULATED_FRAME_NUM_BIT_MASK) << SHARC_ACCUMULATED_FRAME_NUM_BIT_OFFSET);
    packedData.w |= (min(staleFrameNum, SHARC_STALE_FRAME_NUM_BIT_MASK) << SHARC_STALE_FRAME_NUM_BIT_OFFSET);

    bool isValidElement = (staleFrameNum < max(staleFrameNumMax, SHARC_STALE_FRAME_NUM_MIN)) ? true : false;

    if (!isValidElement)
    {
        packedData = make_uint4(0, 0, 0, 0);
#if !SHARC_ENABLE_COMPACTION
        BUFFER_AT_OFFSET(hashMapData.hashEntriesBuffer, entryIndex) = HASH_GRID_INVALID_HASH_KEY;
#endif // !SHARC_ENABLE_COMPACTION
    }

#if SHARC_ENABLE_COMPACTION
    uint validElementNum = WaveActiveCountBits(isValidElement);
    uint validElementMask = WaveActiveBallot(isValidElement);
    bool isMovableElement = isValidElement && ((entryIndex % HASH_GRID_HASH_MAP_BUCKET_SIZE) >= validElementNum);
    uint movableElementIndex = WavePrefixCountBits(isMovableElement);

    if ((entryIndex % HASH_GRID_HASH_MAP_BUCKET_SIZE) >= validElementNum)
    {
        uint writeOffset = 0;
#if !SHARC_DEFERRED_HASH_COMPACTION
        hashMapData.hashEntriesBuffer[entryIndex] = HASH_GRID_INVALID_HASH_KEY;
#endif // !SHARC_DEFERRED_HASH_COMPACTION

        BUFFER_AT_OFFSET(voxelDataBuffer, entryIndex) = make_uint4(0, 0, 0, 0);

        if (isValidElement)
        {
            uint emptySlotIndex = 0;
            while (emptySlotIndex < validElementNum)
            {
                if (((validElementMask >> writeOffset) & 0x1) == 0)
                {
                    if (emptySlotIndex == movableElementIndex)
                    {
                        writeOffset += GetBaseSlot(entryIndex, hashMapData.capacity);
#if !SHARC_DEFERRED_HASH_COMPACTION
                        hashMapData.hashEntriesBuffer[writeOffset] = hashKey;
#endif // !SHARC_DEFERRED_HASH_COMPACTION

                        BUFFER_AT_OFFSET(voxelDataBuffer, writeOffset) = packedData;
                        break;
                    }

                    ++emptySlotIndex;
                }

                ++writeOffset;
            }
        }

#if SHARC_DEFERRED_HASH_COMPACTION
        BUFFER_AT_OFFSET(copyOffsetBuffer, entryIndex) = (writeOffset != 0) ? writeOffset : HASH_GRID_INVALID_CACHE_ENTRY;
#endif // SHARC_DEFERRED_HASH_COMPACTION
    }
    else if (isValidElement)
#endif // SHARC_ENABLE_COMPACTION
    {
        BUFFER_AT_OFFSET(voxelDataBuffer, entryIndex) = packedData;
    }

#if !SHARC_BLEND_ADJACENT_LEVELS
    // Clear buffer entry for the next frame
    //BUFFER_AT_OFFSET(voxelDataBufferPrev, entryIndex) = uint4(0, 0, 0, 0);
#endif // !SHARC_BLEND_ADJACENT_LEVELS
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
