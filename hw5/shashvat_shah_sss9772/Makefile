### ECE-GY 9143 - High Performance Computing for Machine Learning
### Instructor: Parijat Dubey
### Makefile for Part A Assignment 5

SDK_INSTALL_PATH :=  /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
OPTIONS   :=  -O3 
CXX := g++
CXXFLAGS := -Wall -O2

TAR_FILE_NAME  := shashvat_shah_sss9772.tar
EXECS := conv conv_tiled conv_cudnn
# EXECS := vecadd00 vecadd01 matmult00 matmult01 vecAddCpu vecAddB1 vecAddB2 vecAddB3 vecAddB1Unified vecAddB2Unified vecAddB3Unified conv conv_tiled conv_cudnn

all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

#######################################################################
tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.cc *.pdf *.txt
#######################################################################

timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)

#######################################################################
vecaddKernel00.o : vecaddKernel00.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

vecadd00 : vecadd.cu vecaddKernel.h vecaddKernel00.o timer.o
	${NVCC} $< vecaddKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)


######################################################################
vecaddKernel01.o : vecaddKernel01.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

vecadd01 : vecadd.cu vecaddKernel.h vecaddKernel01.o timer.o
	${NVCC} $< vecaddKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)


######################################################################
# Provided Kernel
matmultKernel00.o : matmultKernel00.cu matmultKernel.h 
	${NVCC} $< -c -o $@ $(OPTIONS)

matmult00 : matmult.cu  matmultKernel.h matmultKernel00.o timer.o
	${NVCC} $< matmultKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)


######################################################################
# Expanded Kernel, notice that FOOTPRINT_SIZE is redefined (from 16 to 32)
matmultKernel01.o : matmultKernel01.cu matmultKernel.h
	${NVCC} $< -c -o $@ $(OPTIONS) -DFOOTPRINT_SIZE=32

matmult01 : matmult.cu  matmultKernel.h matmultKernel01.o timer.o
	${NVCC} $< matmultKernel01.o -o $@ $(LIB) timer.o $(OPTIONS) -DFOOTPRINT_SIZE=32
######################################################################
#  Simple vecadd
vecAddCpu: vecAddCpu.o timer.o
	$(NVCC) $(OPTIONS) -o vecAddCpu vecAddCpu.o timer.o

######################################################################

vecAddB1 : vecAddB1.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)

# #######################################################################
vecAddB2 : vecAddB2.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)
# #######################################################################

vecAddB3 : vecAddB3.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)
#######################################################################
# vec add with unified memory access
vecAddB1Unified : vecAddB1Unified.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)
# #######################################################################

vecAddB2Unified : vecAddB2Unified.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)
######################################################################

vecAddB3Unified : vecAddB3Unified.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)
# #######################################################################
# Simple convolution
conv : conv.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS)
#######################################################################
# Tiled Convolution
convTiledKernel.o : convTiledKernel.cu convTiledKernel.h 
	${NVCC} $< -c -o $@ $(OPTIONS)

conv_tiled : conv_tiled.cu  convTiledKernel.h convTiledKernel.o timer.o
	${NVCC} $< convTiledKernel.o -o $@ $(LIB) timer.o $(OPTIONS)

# #######################################################################
# convolution with cudnn kernel optimization
conv_cudnn : conv_cudnn.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS) -lcudnn