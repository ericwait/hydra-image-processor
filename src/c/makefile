#########################################
# Set these for MATLAB/Python versions
#########################################
MATALAB_DIR = /usr/local/MATLAB/R2018b
# NOTE: Currently only builds against python3
PYTHON_VER = python3.5
PYTHON3_INC = /usr/include/$(PYTHON_VER)
NUMPY3_INC = /usr/include/$(PYTHON_VER)
PYTHON_LIB = /usr/lib/$(PYTHON_VER)/config-3.5m-x86_64-linux-gnu

#########################################
# Default files and include dirs
#########################################

OBJ_DIR := Intermediate

CU_FILES := $(wildcard Cuda/*.cu)
CUPP_FILES := $(wildcard Cuda/*.cpp)
MEX_CPP_FILES := $(filter-out Mex/_TemplateMex.cpp, $(wildcard Mex/*.cpp))
PY_CPP_FILES := $(wildcard Python/*.cpp)
SCR_CPP_FILES := $(wildcard ScriptCmds/*.cpp)

CUDA_OBJ = $(addprefix $(OBJ_DIR)/,$(notdir $(CU_FILES:.cu=.o)))
CUDA_CPP_OBJ = $(addprefix $(OBJ_DIR)/,$(notdir $(CUPP_FILES:.cpp=.o)))
MEX_CPP_OBJ = $(addprefix $(OBJ_DIR)/,$(notdir $(MEX_CPP_FILES:.cpp=.o)))
PY_CPP_OBJ = $(addprefix $(OBJ_DIR)/,$(notdir $(PY_CPP_FILES:.cpp=.o)))
SCRPY_CPP_OBJ = $(addprefix $(OBJ_DIR)/Py,$(notdir $(SCR_CPP_FILES:.cpp=.o)))
SCRMEX_CPP_OBJ = $(addprefix $(OBJ_DIR)/Mx,$(notdir $(SCR_CPP_FILES:.cpp=.o)))

CUDA_INC = -I. -I./Cuda
MEX_INC = -I. -I./Mex -I./ScriptCmds -I./external -I$(MATALAB_DIR)/extern/include
PY_INC = -I. -I./Python -I./ScriptCmds -I./external -I$(PYTHON3_INC) -I$(NUMPY3_INC)

MEX_LIB = -L$(MATALAB_DIR)/bin/glnxa64 -lmx -lmex
PY_LIB = -L$(PYTHON_LIB) -l$(PYTHON_VER)

#########################################
# Common parameters
#########################################
# Passed to both Nvidia compiler and gcc/clang
CPP_FLAGS = -std=c++11 -g -D_MWAITXINTRIN_H_INCLUDED

# Passed to gcc/clang and passthrough (--compiler-options) in nvcc
# NOTE: Any warning flags such as '-Wall' should go here
C_FLAGS = -Wall -Wno-sign-compare -fopenmp -fPIC -g

# Pass-through linker arguments (--linker-options) in nvcc linking step
LD_FLAGS = --no-undefined
#########################################
# Nvidia nvcc parameters
#########################################
NVCC_PATH = nvcc
NVCC_FLAGS = $(CPP_FLAGS)
SMODEL = -arch=sm_30
NVCC_INC = -I/usr/include

ifndef COMP
	COMP=clang
endif

ifeq ($(COMP),gcc)
	C_COMPILER = g++
else ifeq ($(COMP),clang)
	C_COMPILER = clang++
endif

OMP_LIB = -lgomp
# NOTE: Some versions of clang use libiomp5 instead of libgomp
#OMP_LIB = -liomp5

#########################################
# gcc/g++/clang parameters
#########################################
GCC_FLAGS = $(CPP_FLAGS) $(C_FLAGS)
GCC_LIBS = $(OMP_LIB)

#########################################
# all projects to build
#########################################
all: matlab python
	@echo Finished compiling

python: HIP.so
	@echo Finished compiling python wrappers

matlab: Mex.mexa64
	@echo Finished compiling matlab wrappers

#########################################
# link it all together
#########################################
HIP.so: $(CUDA_OBJ) $(PY_CPP_OBJ) $(SCRPY_CPP_OBJ) $(CUDA_CPP_OBJ)
	$(NVCC_PATH) $(GCC_LIBS) $(PY_LIB) -shared $^ -o HIP.so $(SMODEL) --linker-options '$(LD_FLAGS)'

Mex.mexa64: $(CUDA_OBJ) $(MEX_CPP_OBJ) $(SCRMX_CPP_OBJ) $(CUDA_CPP_OBJ)
	$(NVCC_PATH) $(GCC_LIBS) $(MEX_LIB) -shared $^ -o Mex.mexa64 $(SMODEL) --linker-options '$(LD_FLAGS)'

#########################################
# compile the cuda library files
#########################################
$(OBJ_DIR)/%.o: Cuda/%.cu
	@mkdir -p $(@D)
	$(NVCC_PATH) $(NVCC_FLAGS) $(CUDA_INC) $(SMODEL) --compiler-options '$(C_FLAGS)' -c $< -o $@ $(NVCC_INC) -dc

$(OBJ_DIR)/%.o: Cuda/%.cpp
	@mkdir -p $(@D)
	$(C_COMPILER) $(GCC_FLAGS) $(CUDA_INC) -c $< -o $@

#########################################
# compile the mex/python wrappers
#########################################
$(OBJ_DIR)/%.o: Mex/%.cpp
	@mkdir -p $(@D)
	$(C_COMPILER) $(GCC_FLAGS) -DMEX_BUILD -DUSE_PROCESS_MUTEX $(MEX_INC) -c $< -o $@

$(OBJ_DIR)/Mx%.o: ScriptCmds/%.cpp
	@mkdir -p $(@D)
	$(C_COMPILER) $(GCC_FLAGS) -DMEX_BUILD -DUSE_PROCESS_MUTEX $(MEX_INC) -c $< -o $@

$(OBJ_DIR)/%.o: Python/%.cpp
	@mkdir -p $(@D)
	$(C_COMPILER) $(GCC_FLAGS) -DPY_BUILD -DUSE_PROCESS_MUTEX $(PY_INC) -c $< -o $@

$(OBJ_DIR)/Py%.o: ScriptCmds/%.cpp
	@mkdir -p $(@D)
	$(C_COMPILER) $(GCC_FLAGS) -DPY_BUILD -DUSE_PROCESS_MUTEX $(PY_INC) -c $< -o $@

#########################################
# files to clean up
#########################################
clean:
	@echo Cleaning
	rm -f $(OBJ_DIR)/*.o
	rm -f Mex.mexa64 HIP.so
