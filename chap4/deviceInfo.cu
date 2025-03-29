#include <stdio.h>

int main() {
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("The computer has %d cuda devices.\n", devCount);
    cudaDeviceProp devProp;
    for (unsigned int i = 0; i < devCount; i++) {
        cudaGetDeviceProperties(&devProp, i);
        printf("name: %s\n", devProp.name);
        printf("totalGlobalMem: %zu\n", devProp.totalGlobalMem);
        printf("sharedMemPerBlock: %zu\n", devProp.sharedMemPerBlock);
        printf("regsPerBlock: %d\n", devProp.regsPerBlock);
        printf("warpSize: %d\n", devProp.warpSize);
        printf("maxThreadsPerBlock: %d\n", devProp.maxThreadsPerBlock);
        printf("maxBlockSize: %d, %d, %d\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
        printf("maxGridSize: %d, %d, %d\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
        printf("multiProcessorCount: %d\n", devProp.multiProcessorCount);
        printf("concurrentKernels: %d\n", devProp.concurrentKernels);
        printf("asyncEngineCount: %d\n", devProp.asyncEngineCount);
        printf("unifiedAddressing: %d\n", devProp.unifiedAddressing);
        printf("memoryBusWidth: %d\n", devProp.memoryBusWidth);
        printf("l2CacheSize: %d\n", devProp.l2CacheSize);
        printf("persistingL2CacheMaxSize: %d\n", devProp.persistingL2CacheMaxSize);
        printf("maxThreadsPerMultiProcessor: %d\n", devProp.maxThreadsPerMultiProcessor);
        printf("streamPrioritiesSupported: %d\n", devProp.streamPrioritiesSupported);
        printf("globalL1CacheSupported: %d\n", devProp.globalL1CacheSupported);
        printf("localL1CacheSupported: %d\n", devProp.localL1CacheSupported);
        printf("sharedMemPerMultiprocessor: %zu\n", devProp.sharedMemPerMultiprocessor);
        printf("regsPerMultiprocessor: %d\n", devProp.regsPerMultiprocessor);
        printf("managedMemory: %d\n", devProp.managedMemory);
        printf("maxBlocksPerMultiProcessor: %d\n", devProp.maxBlocksPerMultiProcessor);
        printf("reservedSharedMemPerBlock: %zu\n", devProp.reservedSharedMemPerBlock);

        int clockRate, memoryClockRate;
        cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, i);
        cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, i);
        printf("clockRate: %d\n", clockRate);
        printf("memoryClockRate: %d\n", memoryClockRate);
    }
}