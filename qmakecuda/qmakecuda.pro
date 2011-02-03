CONFIG += console

message(QMAKE with CUDA)

HEADERS = radixsort.h
SOURCES = radixsort.cpp\
        testradixsort.cpp
		
CUSOURCES = radixsort.cu

CUDA_CC =/usr/local/cuda/bin/nvcc
CUDA_DIR =/usr/local/cuda
CUDA_DEV ="/Developer/GPU Computing/C"
CUDA_COMMON =$$CUDA_DEV/common
CUDA_INC_PATH = $$CUDA_DIR/include

message("nvcc resides in :" $$CUDA_CC)

INCLUDEPATH += $$CUDA_INC_PATH\
                $$CUDA_COMMON/inc
LIBS += -lcudart -lcuda -lcudpp -lcutil_i386

QMAKE_LIBDIR += $$CUDA_DIR/lib\
                $$CUDA_COMMON/lib/darwin\
                $$CUDA_DEV/lib

cuda.name = CUDA
cuda.input = CUSOURCES
cuda.dependency_type = TYPE_C
cuda.variable_out = OBJECTS
cuda.output = ${QMAKE_FILE_IN}$$QMAKE_EXT_OBJ
cuda.commands = $$CUDA_CC -c -arch sm_11 $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_IN} -o ${QMAKE_FILE_OUT} # Note the -O0
QMAKE_EXTRA_COMPILERS += cuda
