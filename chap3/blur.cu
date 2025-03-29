#include <stdio.h>

const int BLUR_SIZE = 1;

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }


__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;
        for (int blurRow=-BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol=-BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                if (curRow >= 0 && curRow < h && curCol >=0 && curCol < w) {
                    pixVal += in[curRow * w + curCol];
                    ++pixels;
                }
            }
        }
        out[row * w + col] = (unsigned char) (pixVal / pixels);
    }
}

void blur(unsigned char * in_h, unsigned char * out_h, int m, int n) {
    int size = m * n * sizeof(unsigned char);
    unsigned char *in_d, *out_d;
    CUDA_CHECK(cudaMalloc((void **) &in_d, size));
    CUDA_CHECK(cudaMalloc((void **) &out_d, size));
    CUDA_CHECK(cudaMemcpy(in_d, in_h, size, cudaMemcpyHostToDevice));
    dim3 dimGrid(ceil(m / 16.0), ceil(n / 16.0), 1);
    dim3 dimBlock(16, 16, 1);
    blurKernel<<<dimGrid, dimBlock>>>(in_d, out_d, m, n);
    CUDA_CHECK(cudaMemcpy(out_h, out_d, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(in_d));
    CUDA_CHECK(cudaFree(out_d));
}

int main() {
    int m = 76;
    int n = 62;
    size_t size = m * n * sizeof(unsigned char);
    unsigned char *in = (unsigned char *)malloc(size);
    unsigned char *out = (unsigned char *)malloc(size);
    for (int i = 0; i < m * n; i++) {
        in[i] = rand() % 256;
    }
    blur(in, out, m, n);
    for (int i = 0; i < m * n; i++) {
        int pixVal = 0;
        int pixels = 0;
        int row = i / m;
        int col = i % m;
        for (int blurRow=-BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol=-BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                if (curRow >= 0 && curRow < n && curCol >=0 && curCol < m) {
                    pixVal += in[curRow * m + curCol];
                    ++pixels;
                }
            }
        }
        if (abs(pixVal / pixels - out[i]) > 1) {
            fprintf(stderr, "Result verification failed at element %d: %d, %d\n", 
                i, pixVal / pixels, out[i]);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");
    free(in);
    free(out);
    return 0;
}