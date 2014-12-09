CONFIG -= app_bundle
CONFIG += release
message(QMAKE with CUDA)
INCLUDEPATH += ../shared
HEADERS       = glWidget.h \
                window.h \
                BezierProgram.h \
                bezier_implement.h \
                ../shared/CudaBase.h \
                ../shared/Base3DView.h \
                ../shared/BaseBuffer.h \
                ../shared/CUDABuffer.h \
                ../shared/CUDAProgram.h
SOURCES       = main.cpp \
                glWidget.cpp \
                window.cpp \
                BezierProgram.cpp \
                ../shared/CudaBase.cpp \
                ../shared/Base3DView.cpp \
                ../shared/BaseBuffer.cpp \
                ../shared/CUDABuffer.cpp \
                ../shared/CUDAProgram.cpp

win32 {
    INCLUDEPATH += D:/usr/boost_1_51_0 /usr/local/include/OpenEXR
    QMAKE_LIBDIR += D:/usr/boost_1_51_0/stage/lib D:/usr/local/openEXR/lib
    CONFIG += console
}
macx {
    INCLUDEPATH += ../../Library/boost_1_55_0
        LIBS += -L../../Library/boost_1_55_0/stage/lib -lboost_date_time -lboost_system -lboost_thread -lboost_filesystem
}
QT           += opengl
LIBS += -L../lib -laphid -lIlmImf -lHalf

CUSOURCES = bezier.cu

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

