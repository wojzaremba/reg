#ifndef COMMON_PTRS_CUH
#define	COMMON_PTRS_CUH

#include "common_gpu.cuh"

const int fsize = 25; 
static void (*func[fsize]) (int, mxArray **, int, const mxArray **) = 
        {CopyToGPU, CopyFromGPU, AddVector, Scale, 
         ActRELU, dActRELU, ActLINEAR, dActLINEAR,
         ConvAct, Reshape, MaxPooling, PrintShape, ActEXP,
         Sum, Max, EltwiseDivideByVector, Mult, ConvResponseNormCrossMap, CleanGPU,
         StartTimer, StopTimer, EltwiseMult, Transpose, Add, Subtract};

#endif	/* COMMON_PTRS_CUH */
