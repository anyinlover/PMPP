#include <stdio.h>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

__global__
void matrixMulKernel(float* M, float* N, float* P, int Width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < Width) {
        for (int row = 0; row < Width; row++) {
            float Pvalue = 0;
            for (int k = 0; k < Width; ++k) {
                Pvalue += M[row * Width + k] * N[k * Width + col];
            }
            P[row * Width + col] = Pvalue;
        }
    }
}

void matrixMul(float* M_h, float* N_h, float* P_h, int Width) {
    int size = Width * Width * sizeof(float);
    float* M_d, *N_d, *P_d;
    CUDA_CHECK(cudaMalloc((void **) &M_d, size));
    CUDA_CHECK(cudaMalloc((void **) &N_d, size));
    CUDA_CHECK(cudaMalloc((void **) &P_d, size));
    CUDA_CHECK(cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice));
    dim3 dimGrid(ceil(Width / 2.0), 1, 1);
    dim3 dimBlock(2, 1, 1);
    matrixMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(M_d));
    CUDA_CHECK(cudaFree(N_d));
    CUDA_CHECK(cudaFree(P_d));
}

int main() {
    int Width = 4;
    size_t size = Width * Width * sizeof(float);
    float *M = (float *)malloc(size);
    float *N = (float *)malloc(size);
    float *P = (float *)malloc(size);
    for (int i = 0; i < Width * Width; i++) {
        M[i] = rand() / (float)RAND_MAX;
        N[i] = rand() / (float)RAND_MAX;
    }
    matrixMul(M, N, P, Width);
    for (int i = 0; i < Width * Width; i++) {
        int row = i / Width;
        int col = i % Width;
        float val = 0;
        for (int j = 0; j < Width; j++) {
            val += M[row * Width + j] * N[j * Width + col];
        }
        if (fabs(val - P[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d: %f, %f\n", i, val, P[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED\n");
    free(M);
    free(N);
    free(P);
    return 0;
}