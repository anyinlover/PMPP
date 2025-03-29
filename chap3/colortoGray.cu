#include <stdio.h>
const int CHANNELS = 3;

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

__global__
void colortoGrayscaleConvertion(unsigned char * Pin,
                            unsigned char * Pout, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int grayOffset = row * width + col;
        int rgbOffset = grayOffset*CHANNELS;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];
        Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

void colortoGray(unsigned char * Pin_h, unsigned char * Pout_h, int m, int n) {
    int size = m * n * sizeof(unsigned char);
    unsigned char *Pin_d, *Pout_d;
    CUDA_CHECK(cudaMalloc((void **) &Pin_d, size * CHANNELS));
    CUDA_CHECK(cudaMalloc((void **) &Pout_d, size));
    CUDA_CHECK(cudaMemcpy(Pin_d, Pin_h, size * CHANNELS, cudaMemcpyHostToDevice));
    dim3 dimGrid(ceil(m / 16.0), ceil(n / 16.0), 1);
    dim3 dimBlock(16, 16, 1);
    colortoGrayscaleConvertion<<<dimGrid, dimBlock>>>(Pin_d, Pout_d, m, n);
    CUDA_CHECK(cudaMemcpy(Pout_h, Pout_d, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(Pin_d));
    CUDA_CHECK(cudaFree(Pout_d));
}

int main() {
    int m = 76;
    int n = 62;
    size_t size = m * n * sizeof(unsigned char);
    unsigned char *Pin = (unsigned char *)malloc(size * CHANNELS);
    unsigned char *Pout = (unsigned char *)malloc(size);
    for (int i = 0; i < m * n * CHANNELS; i++) {
        Pin[i] = rand() % 256;
    }
    colortoGray(Pin, Pout, m, n);
    for (int i = 0; i < m * n; i++) {
        unsigned char out = 0.21f * Pin[i * CHANNELS] + 0.71f * Pin[i * CHANNELS + 1] + 0.07f * Pin[i * CHANNELS + 2];
        // gpu and cpu may get different result because of float calc
        if (abs(out - Pout[i]) > 1) {
            fprintf(stderr, "Result verification failed at element %d: %d, %d, %d, %d, %d\n", 
                i, Pin[i * CHANNELS], Pin[i * CHANNELS + 1], Pin[i * CHANNELS + 2], Pout[i], out);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");
    free(Pin);
    free(Pout);
    return 0;
}