CONFIG -= app_bundle
CONFIG += release
message(QMAKE with CUDA)
INCLUDEPATH += ../shared
HEADERS = ../shared/Vector3F.h \
    ../shared/Matrix44F.h \
    ../shared/BaseCamera.h \
    ../shared/shapeDrawer.h \
    ../shared/Polytode.h \
    ../shared/Vertex.h \
    ../shared/Facet.h \
    ../shared/GeoElement.h \
    ../shared/Edge.h \
    ../shared/BaseMesh.h \
    ../shared/HemisphereMesh.h \
    ../shared/BaseBuffer.h \
    ../shared/CUDABuffer.h \
    ../shared/CUDAProgram.h \
    glwidget.h \
    window.h \
    HemisphereProgram.h \
    BRDFProgram.h \
    hemisphere_implement.h \
    Lambert.h \
    lambert_implement.h \
    Phong.h \
    phong_implement.h
    
SOURCES = ../shared/Vector3F.cpp \
    ../shared/Matrix44F.cpp \
    ../shared/BaseCamera.cpp \
    ../shared/shapeDrawer.cpp \
    ../shared/Polytode.cpp \
    ../shared/Vertex.cpp \
    ../shared/Facet.cpp \
    ../shared/GeoElement.cpp \
    ../shared/Edge.cpp \
    ../shared/BaseMesh.cpp \
    ../shared/HemisphereMesh.cpp \
    ../shared/BaseBuffer.cpp \
    ../shared/CUDABuffer.cpp \
    ../shared/CUDAProgram.cpp \
    glwidget.cpp \
    main.cpp \
    window.cpp \
    HemisphereProgram.cpp \
    BRDFProgram.cpp \
    Lambert.cpp \
    Phong.cpp
    
CUSOURCES = hemisphere.cu \
            lambert.cu \
            phong.cu

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

QT += opengl
win32:CONFIG += console
