TEMPLATE = app
CONFIG -= app_bundle

# CONFIG          +=    console
TARGET = compact
message(QMAKE with CUDA)
HEADERS = addressbook.h \
            compact_implement.h
SOURCES = main.cpp \
    addressbook.cpp
CUSOURCES = compact.cu \
            scan.cu
CUDA_CC = /usr/local/cuda/bin/nvcc
CUDA_DIR = /usr/local/cuda
CUDA_DEV = "/Developer/GPU Computing/C"
CUDA_COMMON = $$CUDA_DEV/common
CUDA_INC_PATH = $$CUDA_DIR/include
CUDA_SHARED = "/Developer/GPU Computing/shared"
message("nvcc resides in :" $$CUDA_CC)
INCLUDEPATH += $$CUDA_INC_PATH \
    $$CUDA_COMMON/inc \
    $$CUDA_SHARED/inc
LIBS += -lcuda \
    -lcudart \
    -lcutil_i386 \
    -lshrutil_i386


# QMAKE_LFLAGS += -Xlinker -rpath $$CUDA_DIR/lib
QMAKE_LIBDIR += $$CUDA_DIR/lib \
    $$CUDA_COMMON/lib/darwin \
    $$CUDA_DEV/lib \
    $$CUDA_SHARED/lib
cuda.name = CUDA
cuda.input = CUSOURCES
cuda.dependency_type = TYPE_C
cuda.variable_out = OBJECTS
cuda.output = ${QMAKE_FILE_IN}$$QMAKE_EXT_OBJ
cuda.commands = $$CUDA_CC \
    -c \
    -m32 \
    -arch sm_11 \
    $$join(INCLUDEPATH,'" -I "','-I "','"') \
    ${QMAKE_FILE_IN} \
    -o \
    ${QMAKE_FILE_OUT} # Note the -O0
QMAKE_EXTRA_COMPILERS += cuda
RESOURCES += misc.qrc
