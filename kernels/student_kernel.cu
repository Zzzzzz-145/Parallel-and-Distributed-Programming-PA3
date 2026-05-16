#include "../math_utils.h"
#include <stdio.h>

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

__global__ void StudentKernel(int M, int N, int K, float alpha,
                              float *A, float *B, float beta, float *C) {
    // 保持 V3 原本 shared memory layout
    // 這裡先不要改成 A transpose，也先不要只 padding B
    __shared__ float As[BM][BK + 4];
    __shared__ float Bs[BK][BN ];

    int tid = threadIdx.x;        // 0..255

    int warpId = tid >> 5;        // 0..7
    int lane   = tid & 31;        // 0..31
    
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;
    
    // explicit warp-level tiling
    // each warp computes a 16x32 sub-tile
    int warpTileRow = (warpId >> 1) * 16;  // 0, 16, 32, 48
    int warpTileCol = (warpId & 1) * 32;   // 0 or 32
    
    // within each warp:
    // laneRow = 0..3
    // laneCol = 0..7
    int laneRow = lane >> 3;
    int laneCol = lane & 7;
    
    int localRow = warpTileRow + laneRow * TM;
    int localCol = warpTileCol + laneCol * TN;

    float acc[TM][TN];

#pragma unroll
    for (int i = 0; i < TM; i++) {
#pragma unroll
        for (int j = 0; j < TN; j++) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---------------------------------------------------------------------
        // float4 load A tile
        //
        // A tile: BM x BK = 64 x 16 = 1024 floats
        // float4 groups: 1024 / 4 = 256
        // 256 threads => each thread loads exactly one float4
        //
        // A is row-major, so A[row][k0 + c ... k0 + c+3] is contiguous.
        // ---------------------------------------------------------------------
        {
            int vecId = tid;               // 0..255
            int r = vecId / (BK / 4);      // 0..63
            int c4 = vecId % (BK / 4);     // 0..3
            int c = c4 * 4;                // 0, 4, 8, 12

            const float4 a4 = reinterpret_cast<const float4*>(
                A + (blockRow + r) * K + (k0 + c)
            )[0];

            As[r][c + 0] = a4.x;
            As[r][c + 1] = a4.y;
            As[r][c + 2] = a4.z;
            As[r][c + 3] = a4.w;
        }

        // ---------------------------------------------------------------------
        // float4 load B tile
        //
        // B tile: BK x BN = 16 x 64 = 1024 floats
        // float4 groups: 1024 / 4 = 256
        // 256 threads => each thread loads exactly one float4
        //
        // B is row-major, so B[k0 + r][col ... col+3] is contiguous.
        // ---------------------------------------------------------------------
        {
            int vecId = tid;               // 0..255
            int r = vecId / (BN / 4);      // 0..15
            int c4 = vecId % (BN / 4);     // 0..15
            int c = c4 * 4;                // 0, 4, 8, ..., 60

            const float4 b4 = reinterpret_cast<const float4*>(
                B + (k0 + r) * N + (blockCol + c)
            )[0];

            Bs[r][c + 0] = b4.x;
            Bs[r][c + 1] = b4.y;
            Bs[r][c + 2] = b4.z;
            Bs[r][c + 3] = b4.w;
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

    // 保持 V3 原本 scalar C store
    // 先不要加入 float4 C load/store，這樣才能單獨測 A/B float4 load 的效果。
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        float4 oldC = reinterpret_cast<const float4*>(
            C + (globalRow + i) * N + globalCol
        )[0];
    
        float4 out;
        out.x = alpha * acc[i][0] + beta * oldC.x;
        out.y = alpha * acc[i][1] + beta * oldC.y;
        out.z = alpha * acc[i][2] + beta * oldC.z;
        out.w = alpha * acc[i][3] + beta * oldC.w;
    
        reinterpret_cast<float4*>(
            C + (globalRow + i) * N + globalCol
        )[0] = out;
    }
}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
    dim3 block(256);
    dim3 grid(N / BN, M / BM);

    StudentKernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}