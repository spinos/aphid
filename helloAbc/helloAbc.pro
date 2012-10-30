ABC_SRC = D:/usr/Alembic/lib/Alembic
HDF5_ROOT = D:/usr/hdf5

INCLUDEPATH += D:/usr/Alembic/lib \
                D:/usr/local/include \
                D:/usr/local/include/boost/tr1/tr1 \
                $$HDF5_ROOT/include \
                ../shared
                
QMAKE_LIBDIR += $$HDF5_ROOT/lib \
                D:/usr/local/lib64

HEADERS       = ../shared/ALFile.h
                
SOURCES       = ../shared/ALFile.cpp \
		main.cpp
                
DEFINES += OPENEXR_DLL
                
LIBS += -lhdf5 -lhdf5_hl -lszip -lHalf -lIex -lIlmImf -lImath -lIlmThread -lalembic

win32 {
CONFIG += console
}
