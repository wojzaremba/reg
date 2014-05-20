#ifndef COMMON_GPU_CUH
#define	COMMON_GPU_CUH

#include <stdio.h>
#include <algorithm>
#include <vector>

using namespace std;

#include "nvmatrix.cuh"
#include "nvmatrix_kernels.cuh"
#include "nvmatrix_operators.cuh"
#include "common_params.cuh"


void MaxPooling(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void MaxPoolingUndo(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void ConvAct(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void ConvActUndo(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void ConvResponseNormCrossMap(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void ConvResponseNormCrossMapUndo(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void cleanUp();
void CleanGPU(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void Reshape(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void ActEXP(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void ActRELU(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void dActRELU(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void dActLINEAR(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void ActLINEAR(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void AddVector(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void Add(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void Subtract(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void Scale(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void Mult(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void EltwiseMult(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void Sum(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void Max(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void Transpose(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void EltwiseDivideByVector(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void PrintShape(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void CopyFromGPU(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
std::vector<NVMatrix*>* Allocate();
void CopyToGPU(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void StartTimer(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
void StopTimer(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

#include "common_ptrs.cuh"

#endif	/* COMMON_GPU_CUH */
