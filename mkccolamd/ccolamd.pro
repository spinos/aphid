TEMPLATE = lib
TARGET = ccolamd
CONFIG += staticlib thread release
CONFIG -= qt
CCOLAMD_SRC = D:/usr/CCOLAMD/Source

INCLUDEPATH += D:/usr/CCOLAMD/Include D:/usr/SuiteSparse_config

SOURCES       = $$CCOLAMD_SRC/ccolamd.c \
                $$CCOLAMD_SRC/ccolamd_global.c  

                
DEFINES += NDEBUG
                



