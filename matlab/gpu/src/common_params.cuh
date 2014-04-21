#ifndef COMMON_PARAMS_CUH
#define	COMMON_PARAMS_CUH


#include "nvmatrix.cuh"
#include "nvmatrix_kernels.cuh"
#include "nvmatrix_operators.cuh"

static cudaEvent_t start_event, end_event;

static std::vector<NVMatrix*> *p = NULL;

NVMatrix* getMatrix(const mxArray *prhs) {
	int idx = (int)mxGetScalar(prhs);
	assert_(idx < (*p).size());
	assert_((*p)[idx] != NULL);
	return (*p)[idx];
}

#define GET3()       assert_((nrhs == 4) && (nlhs == 0)); \
   	             NVMatrix* a = getMatrix(prhs[1]); \
	             NVMatrix* b = getMatrix(prhs[2]); \
	             NVMatrix* c = getMatrix(prhs[3]);

#define GET1X1(TYPE) assert_((nrhs == 4) && (nlhs == 0)); \
		     NVMatrix* a = getMatrix(prhs[1]); \
	             TYPE b = (TYPE)mxGetScalar(prhs[2]); \
	             NVMatrix* c = getMatrix(prhs[3]);
                 
#define GET2()       assert_((nrhs == 3) && (nlhs == 0)); \
	             NVMatrix* a = getMatrix(prhs[1]); \
	             NVMatrix* b = getMatrix(prhs[2]); 

#endif	/* COMMON_PARAMS_CUH */
