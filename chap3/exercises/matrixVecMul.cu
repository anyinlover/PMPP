__global__
void matrixVecMulKernel(float* A, float* B, float* C, int Width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < Width) {
        float Pvalue = 0;
        for (int col = 0; col < Width; ++col) {
            Pvalue += B[row * Width + col] * C[col];
        }
        A[row] = Pvalue;
    }
}
