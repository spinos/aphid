TEMPLATE = lib
TARGET = colamd
CONFIG += staticlib thread release
CONFIG -= qt
COLAMD_SRC = D:/usr/COLAMD/Source

INCLUDEPATH += D:/usr/COLAMD/Include D:/usr/SuiteSparse_config

SOURCES       = $$COLAMD_SRC/colamd.c \
                $$COLAMD_SRC/colamd_global.c  

                
DEFINES += NDEBUG
                



