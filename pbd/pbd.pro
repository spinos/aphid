INCLUDEPATH += ../shared ../ ./
LIBS += -L../lib -laphid -lIlmImf -lHalf
macx {
    INCLUDEPATH += $(HOME)/Library/boost_1_55_0 \
        /usr/local/include/bullet
    LIBS += -lboost_date_time -lboost_system -lboost_thread -lboost_filesystem 
}
win32 {
    INCLUDEPATH += D:/usr/boost_1_51_0 \
                    D:/usr/local/openEXR/include
    QMAKE_LIBDIR += D:/usr/boost_1_51_0/stage/lib \
                    D:/usr/local/openEXR/lib
    QMAKE_LFLAGS_RELEASE += /NODEFAULTLIB:libcmt  /NODEFAULTLIB:libcpmt
}

HEADERS       = glwidget.h \
                window.h \
                ../shared/Base3DView.h \
                 ../shared/GLHUD.h \
                 ../shared/BaseBuffer.h \
                ../shared/CUDABuffer.h \
                ../shared/CUDAProgram.h \
                 BoxProgram.h
SOURCES       = ../shared/Base3DView.cpp \
                ../shared/GLHUD.cpp \
                ../shared/BaseBuffer.cpp \
                ../shared/CUDABuffer.cpp \
                ../shared/CUDAProgram.cpp \
                glwidget.cpp \
                main.cpp \
                window.cpp \
                BoxProgram.cpp
QT           += opengl
win32:CONFIG += console
DESTDIR = ./

CUSOURCES = box.cu
macx {
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
QMAKE_LIBDIR += $$CUDA_DIR/lib \
    $$CUDA_COMMON/lib/darwin \
    $$CUDA_DEV/lib \
    $$CUDA_SHARED/lib
}
win32 {
CUDA_CC =D:/usr/cuda4/v4.0/bin/nvcc.exe
CUDA_DIR = D:/usr/cuda4/v4.0
CUDA_DEV = "D:/usr/cuda4_sdk/C"
CUDA_COMMON = $$CUDA_DEV/common
CUDA_INC_PATH = $$CUDA_DIR/include
message("nvcc resides in :" $$CUDA_CC)
INCLUDEPATH += $$CUDA_INC_PATH \
                $$CUDA_COMMON/inc

QMAKE_LIBDIR += $$CUDA_DIR/lib/x64 \
                $$CUDA_COMMON/lib/x64
QMAKE_LFLAGS_RELEASE += /NODEFAULTLIB:libcmt  /NODEFAULTLIB:libcpmt
LIBS += -lcuda -lcudart
}

cuda.name = CUDA
cuda.input = CUSOURCES
cuda.dependency_type = TYPE_C
cuda.variable_out = OBJECTS
cuda.output = ${QMAKE_FILE_IN}$$QMAKE_EXT_OBJ
cuda.commands = $$CUDA_CC \
    -c \
    -m64 \
    -arch sm_11 \
    $$join(INCLUDEPATH,'" -I "','-I "','"') \
    ${QMAKE_FILE_IN} \
    -o \
    ${QMAKE_FILE_OUT} # Note the -O0
QMAKE_EXTRA_COMPILERS += cuda
