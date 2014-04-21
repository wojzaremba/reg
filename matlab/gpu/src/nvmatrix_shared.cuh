#ifndef NVMATRIX_SHARED_H_
#define NVMATRIX_SHARED_H_


#include <curand_kernel.h>

#if defined(_WIN64) || defined(_WIN32)
#define unsigned int unsigned int
#endif

#define NUM_BLOCKS_MAX                      65535

#define NUM_RND_BLOCKS                      96
#define NUM_RND_THREADS_PER_BLOCK           128
#define NUM_RND_STREAMS                     (NUM_RND_BLOCKS * NUM_RND_THREADS_PER_BLOCK)

/*
 * Default grid/block sizes for the various functions.
 */
#define ADD_BLOCK_SIZE                      16

#define NUM_TILE_BLOCKS                     4096
#define NUM_TILE_THREADS_PER_BLOCK          512

#define ELTWISE_THREADS_X                   32
#define ELTWISE_THREADS_Y                   8

#define NUM_SUM_COLS_THREADS_PER_BLOCK      256

#define AGG_SHORT_ROWS_THREADS_X            32
#define AGG_SHORT_ROWS_THREADS_Y            8
#define AGG_SHORT_ROWS_LOOPS_Y              32

#define DP_BLOCKSIZE                        512
#define CPUSUM_MAX                          4096

#define ADD_VEC_THREADS_X                   64
#define ADD_VEC_THREADS_Y                   4

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

#define MYMAX(a, b) ((a) > (b) ? (a) : (b))

#ifndef MUL24 // legacy
#define MUL24(x,y) ((x) * (y))
#endif

#define AWR_NUM_THREADS           256
#define WARP_SIZE                 32
#define AWR_NUM_WARPS             AWR_NUM_THREADS / WARP_SIZE 
#define AWR_LOG_NUM_THREADS       8
#define LOG_WARP_SIZE             5
#define AWR_LOG_NUM_WARPS         3

__global__ void kTile(const float* src, float* tgt, const unsigned int srcWidth, const unsigned int srcHeight, const unsigned int tgtWidth, const unsigned int tgtHeight);
__global__ void kDotProduct_r(float* a, float* b, float* target, const unsigned int numCols, const unsigned int numElements);
__global__ void kSetupCurand(curandState *state, unsigned long long seed);

#endif /* NVMATRIX_SHARED_H_ */
