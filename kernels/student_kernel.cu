#include "../math_utils.h"
#include <stdio.h>

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

__global__ void StudentKernel(int M, int N, int K, float alpha,
                              float *A, float *B, float beta, float *C) {
    __shared__ float As[BM][BK + 1];   // +1 padding helps reduce bank conflicts
    __shared__ float Bs[BK][BN + 1];

    int tx = threadIdx.x;  // 0..15
    int ty = threadIdx.y;  // 0..15

    int tid = ty * blockDim.x + tx;    // 0..255

    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    int localRow = ty * TM;  // thread's starting row inside C tile
    int localCol = tx * TN;  // thread's starting col inside C tile

    float acc[TM][TN];

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile: BM x BK = 64 x 16 = 1024 floats
        // 256 threads, each loads 4 A elements
        for (int idx = tid; idx < BM * BK; idx += blockDim.x * blockDim.y) {
            int r = idx / BK;
            int c = idx % BK;
            As[r][c] = A[(blockRow + r) * K + (k0 + c)];
        }

        // Load B tile: BK x BN = 16 x 64 = 1024 floats
        // 256 threads, each loads 4 B elements
        for (int idx = tid; idx < BK * BN; idx += blockDim.x * blockDim.y) {
            int r = idx / BN;
            int c = idx % BN;
            Bs[r][c] = B[(k0 + r) * N + (blockCol + c)];
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float aFrag[TM];
            float bFrag[TN];

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                aFrag[i] = As[localRow + i][kk];
            }

            #pragma unroll
            for (int j = 0; j < TN; j++) {
                bFrag[j] = Bs[kk][localCol + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    acc[i][j] = fmaf(aFrag[i], bFrag[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    int globalRow = blockRow + localRow;
    int globalCol = blockCol + localCol;

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int row = globalRow + i;
            int col = globalCol + j;
            C[row * N + col] = alpha * acc[i][j] + beta * C[row * N + col];
        }
    }
}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
    dim3 block(BN / TN, BM / TM);  // 16 x 16 = 256 threads
    dim3 grid(N / BN, M / BM);

    StudentKernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}