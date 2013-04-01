TEMPLATE = lib
TARGET = camd
CONFIG += staticlib thread release
CONFIG -= qt
CAMD_SRC = D:/usr/CAMD/Source

INCLUDEPATH += D:/usr/CAMD/Include D:/usr/SuiteSparse_config

SOURCES       = $$CAMD_SRC/camd_aat.c \
                $$CAMD_SRC/camd_1.c \
                $$CAMD_SRC/camd_2.c \
                $$CAMD_SRC/camd_dump.c \
                $$CAMD_SRC/camd_postorder.c \
                $$CAMD_SRC/camd_defaults.c \
                $$CAMD_SRC/camd_order.c \
                $$CAMD_SRC/camd_control.c \
                $$CAMD_SRC/camd_info.c \
                $$CAMD_SRC/camd_valid.c \
                $$CAMD_SRC/camd_preprocess.c \
                $$CAMD_SRC/camd_global.c   

                
DEFINES += NDEBUG
                



