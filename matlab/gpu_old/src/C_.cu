#include <stdio.h>
#include "mex.h"
#include "common_gpu.cuh" // Contains all the symbols.

// Entry point.
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	assert_(nrhs >= 1);
  mexAtExit(cleanUp);
	p = Allocate();
	int fid = (int)mxGetScalar(prhs[0]);
	assert_(fid < fsize);
	(*func[fid])(nlhs, plhs, nrhs, prhs);
}
