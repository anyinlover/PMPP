#include <stdio.h>
#include <curand_kernel.h>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    CUDA_CHECK(cudaMalloc((void **) &A_d, size));
    CUDA_CHECK(cudaMalloc((void **) &B_d, size));
    CUDA_CHECK(cudaMalloc((void **) &C_d, size));

    CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));
    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);
    CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}
int main() {
    int N = 50000;
    size_t size = N * sizeof(float);
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);

    for (int i = 0; i < N; ++i) {
        A[i] = rand() / (float)RAND_MAX;
        B[i] = rand() / (float)RAND_MAX;
    }
    vecAdd(A, B, C, N); 
    
    for (int i = 0; i < N; ++i) {
        if (fabs(A[i] + B[i] - C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d: %f, %f, %f\n", i, A[i], B[i], C[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED\n");
    free(A);
    free(B);
    free(C);
    return 0;
}