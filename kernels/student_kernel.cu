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
    __shared__ float As[2][BM][BK + 4];
    __shared__ float Bs[2][BK][BN];

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

// -----------------------------------------------------------------------------
// Double-buffered shared memory pipeline
// -----------------------------------------------------------------------------
// Buffer 0: current tile
// Buffer 1: next tile
//
// Strategy:
// 1. Load tile 0 into shared buffer 0.
// 2. For each k0:
//    - issue global load for next tile into registers
//    - compute current shared tile
//    - store next registers into the other shared buffer
//    - sync and swap buffers
// -----------------------------------------------------------------------------

int readBuf = 0;

// preload k0 = 0 into shared buffer 0
{
    int vecId = tid;

    // A tile: BM x BK = 64 x 16 = 1024 floats = 256 float4
    int ar = vecId / (BK / 4);      // 0..63
    int ac4 = vecId % (BK / 4);     // 0..3
    int ac = ac4 * 4;               // 0,4,8,12

    const float4 a4 = reinterpret_cast<const float4*>(
        A + (blockRow + ar) * K + ac
    )[0];

    As[0][ar][ac + 0] = a4.x;
    As[0][ar][ac + 1] = a4.y;
    As[0][ar][ac + 2] = a4.z;
    As[0][ar][ac + 3] = a4.w;

    // B tile: BK x BN = 16 x 64 = 1024 floats = 256 float4
    int br = vecId / (BN / 4);      // 0..15
    int bc4 = vecId % (BN / 4);     // 0..15
    int bc = bc4 * 4;               // 0,4,...,60

    const float4 b4 = reinterpret_cast<const float4*>(
        B + br * N + (blockCol + bc)
    )[0];

    Bs[0][br][bc + 0] = b4.x;
    Bs[0][br][bc + 1] = b4.y;
    Bs[0][br][bc + 2] = b4.z;
    Bs[0][br][bc + 3] = b4.w;
}

__syncthreads();

for (int k0 = 0; k0 < K; k0 += BK) {
    int nextK = k0 + BK;
    int writeBuf = readBuf ^ 1;
    bool hasNext = nextK < K;

    // -------------------------------------------------------------------------
    // Issue next tile global loads into registers.
    // These values are independent of current compute, so compiler may schedule
    // the global loads ahead and overlap part of the latency with FFMA work.
    // -------------------------------------------------------------------------
    float4 nextA4;
    float4 nextB4;

    int vecId = tid;

    int ar = vecId / (BK / 4);
    int ac4 = vecId % (BK / 4);
    int ac = ac4 * 4;

    int br = vecId / (BN / 4);
    int bc4 = vecId % (BN / 4);
    int bc = bc4 * 4;

    if (hasNext) {
        nextA4 = reinterpret_cast<const float4*>(
            A + (blockRow + ar) * K + (nextK + ac)
        )[0];

        nextB4 = reinterpret_cast<const float4*>(
            B + (nextK + br) * N + (blockCol + bc)
        )[0];
    }

    // -------------------------------------------------------------------------
    // Compute current tile from shared memory
    // -------------------------------------------------------------------------
#pragma unroll
    for (int kk = 0; kk < BK; kk++) {
        float aFrag[TM];
        float bFrag[TN];

#pragma unroll
        for (int i = 0; i < TM; i++) {
            aFrag[i] = As[readBuf][localRow + i][kk];
        }

#pragma unroll
        for (int j = 0; j < TN; j++) {
            bFrag[j] = Bs[readBuf][kk][localCol + j];
        }

#pragma unroll
        for (int i = 0; i < TM; i++) {
#pragma unroll
            for (int j = 0; j < TN; j++) {
                acc[i][j] = fmaf(aFrag[i], bFrag[j], acc[i][j]);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Store prefetched next tile from registers into the other shared buffer
    // -------------------------------------------------------------------------
    if (hasNext) {
        As[writeBuf][ar][ac + 0] = nextA4.x;
        As[writeBuf][ar][ac + 1] = nextA4.y;
        As[writeBuf][ar][ac + 2] = nextA4.z;
        As[writeBuf][ar][ac + 3] = nextA4.w;

        Bs[writeBuf][br][bc + 0] = nextB4.x;
        Bs[writeBuf][br][bc + 1] = nextB4.y;
        Bs[writeBuf][br][bc + 2] = nextB4.z;
        Bs[writeBuf][br][bc + 3] = nextB4.w;
    }

    __syncthreads();

    readBuf ^= 1;
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