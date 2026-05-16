#include "../math_utils.h"
#include <stdio.h>

// =============================================================================
// HW4: CUDA Matmul Optimization
//
// C = alpha * (A @ B) + beta * C
// A: M x K
// B: K x N
// C: M x N
// row-major
// =============================================================================

__global__ void StudentKernel(int M, int N, int K, float alpha,
                              float *A, float *B, float beta, float *C) {
    // threadIdx.x 對應 column，讓 C 的寫入比較連續
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < K; k++) {
            float a = A[row * K + k];   // A[row][k]
            float b = B[k * N + col];   // B[k][col]
            sum += a * b;
        }

        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
    dim3 block(16, 16);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    StudentKernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}