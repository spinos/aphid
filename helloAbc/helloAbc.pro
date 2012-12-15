HEADERS       = ../shared/ALFile.h \
                ../shared/ALTransform.h \
                ../shared/ALMesh.h \
                ../shared/ALIFile.h \
                ../shared/ALITransform.h \
                ../shared/ALIMesh.h 

SOURCES       = ../shared/ALFile.cpp \
                ../shared/ALTransform.cpp \
                ../shared/ALMesh.cpp \
                ../shared/ALIFile.cpp \
                ../shared/ALITransform.cpp \
                ../shared/ALIMesh.cpp \
                main.cpp

win32 {
    CONFIG += console
    ABC_SRC = D:/usr/Alembic/lib
    BOOST_SRC = D:/usr/local/include
    HDF5_ROOT = D:/usr/hdf5
    INCLUDEPATH += D:/usr/local/include
}

macx {
    ABC_SRC = /Users/jianzhang/Library/alembic-1.0.5/lib
    BOOST_SRC = /Users/jianzhang/Library/boost_1_44_0
    HDF5_ROOT = /Users/jianzhang/Library/hdf5
    INCLUDEPATH += /usr/local/include/OpenEXR
}

INCLUDEPATH += $$ABC_SRC \
                $$BOOST_SRC \
                $$BOOST_SRC/boost/tr1/tr1 \
                $$HDF5_ROOT/include \
                ../shared
                
QMAKE_LIBDIR += $$HDF5_ROOT/lib

LIBS += -lhdf5 -lhdf5_hl -lHalf -lIex -lIlmImf -lImath -lIlmThread

win32 {
    DEFINES += OPENEXR_DLL
    QMAKE_LIBDIR += D:/usr/local/lib64
    LIBS += -lszip -lalembic
}

macx {
    CONFIG -= app_bundle
    QMAKE_LIBDIR += $$ABC_SRC
    LIBS += -lAlembicAbc -lAlembicAbcCoreAbstract -lAlembicAbcCoreHDF5 -lAlembicAbcGeom -lAlembicUtil
}
