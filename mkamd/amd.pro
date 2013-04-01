TEMPLATE = lib
TARGET = amd
CONFIG += staticlib thread release
CONFIG -= qt
AMD_SRC = D:/usr/AMD/Source

INCLUDEPATH += D:/usr/AMD/Include D:/usr/SuiteSparse_config

SOURCES       = $$AMD_SRC/amd_aat.c \
                $$AMD_SRC/amd_1.c \
                $$AMD_SRC/amd_2.c \
                $$AMD_SRC/amd_dump.c \
                $$AMD_SRC/amd_postorder.c \
                $$AMD_SRC/amd_post_tree.c \
                $$AMD_SRC/amd_defaults.c \
                $$AMD_SRC/amd_order.c \
                $$AMD_SRC/amd_control.c \
                $$AMD_SRC/amd_info.c \
                $$AMD_SRC/amd_valid.c \
                $$AMD_SRC/amd_preprocess.c \
                $$AMD_SRC/amd_global.c   

                
DEFINES += NDEBUG
                



