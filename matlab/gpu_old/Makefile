include ./findcudalib.mk

UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
	CUDA_PATH   := /usr/local/pkg/cuda/5.5/cuda
	MATLAB_LIB  := -I/opt/matlab/extern/include
	NVCC        := $(CUDA_PATH)/bin/nvcc $(MATLAB_LIB) -L$(CUDA_PATH)/lib64 -L. -lcudart -lcublas -arch=sm_30 --compiler-options -fPIC --output-file
	MEX 	    := mex -DUNIX $(MATLAB_LIB) -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -L.
	EXT	    := so
else
	CUDA_PATH   := /Developer/NVIDIA/CUDA-5.5
	MATLAB_LIB  := -I/Applications/MATLAB_R2013a_Student.app/extern/include/
	NVCC        := $(CUDA_PATH)/bin/nvcc -ccbin /usr/bin/clang -shared $(MATLAB_LIB) -L$(CUDA_PATH)/lib -L. -arch=sm_30 --compiler-options -fPIC -Xcompiler -stdlib=libstdc++ --output-file
	MEX 	    := /Applications/MATLAB_R2013a_Student.app/bin/mex $(MATLAB_LIB) -L$(CUDA_PATH)/lib -L. 
	EXT	    := dylib
endif


################################################################################

all: 
	make libcommon_gpu.$(EXT)
	rm -rf C_.*
	make C_.cpp
	$(MEX) C_.cpp -o C_ -lcudart -lcublas -lcommon_gpu
	mv C_.mex* ../

%.cpp: src/%.cu
	$(NVCC) $@ --cuda $<

%.o: %.cpp
	g++ -shared -c $<

lib%.so: %.cpp
	g++ -shared -o $@ -fPIC $<

lib%.dylib: %.cpp
	g++ -dynamiclib -undefined suppress -flat_namespace $< -o $@

clean:
	rm -rf *.cpp
	rm -rf *.o
	rm -rf *.mex*
	rm -rf ../*.mex*
	rm -rf *.dylib
	rm -rf *.so
