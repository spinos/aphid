TARGET = simple
CONFIG += thread release
CONFIG -= qt
DESTDIR = ./

win32 {
    Aphid = D:/aphid
}
mac {
    Aphid = $$(HOME)/aphid
}

INCLUDEPATH += ./ ../shared ../lapl ../catmullclark ../easymodel ../../ofl/opium ../fit

SOURCES       = main.cpp
macx {
    INCLUDEPATH += ../../Library/boost_1_44_0 \
                    ../../Library/hdf5/include \
                    /usr/local/include/OpenEXR 
    QMAKE_LIBDIR += ../../Library/boost_1_44_0/stage/lib \
                    ../../Library/hdf5/lib
    LIBS += -lboost_date_time -lboost_thread -lboost_filesystem -lboost_system -framework libxml
}
win32 {
    INCLUDEPATH += D:/usr/local/include D:/ofl/shared \
                   D:/usr/hdf5/include \
                   D:/usr/libxml2x64/include \
                   D:/usr/arnoldSDK/arnold4014/include
    QMAKE_LIBDIR += D:/usr/local/lib64 
    LIBS += -LD:/usr/libxml2x64/lib -llibxml2 \
            -LD:/usr/hdf5/lib -lszip \
            -LD:/usr/arnoldSDK/arnold4014/lib -lai
    DEFINES += OPENEXR_DLL NDEBUG NOMINMAX
    CONFIG += console
}
LIBS += -lIlmImf -lHalf -lhdf5 -lhdf5_hl


