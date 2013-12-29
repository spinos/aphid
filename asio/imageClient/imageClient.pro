win32 {
    Aphid = D:/aphid
}
mac {
    Aphid = $$(HOME)/aphid
}
CONFIG -= qt
INCLUDEPATH += ./ ../shared ../lapl ../catmullclark ../easymodel ../../ofl/opium ../fit
SOURCES       = main.cpp
                
LIBS += -lIlmImf -lHalf
macx {
    INCLUDEPATH += $$(HOME)/Library/boost_1_44_0 \
                    /usr/local/include/OpenEXR \
                    $$(HOME)/Library/eigen2
    QMAKE_LIBDIR += ../../Library/boost_1_44_0/stage/lib \
                    ../../Library/hdf5/lib
    LIBS += -lboost_date_time -lboost_thread -lboost_filesystem -lboost_system
    CONFIG -= app_bundle
}
win32 {
    HEADERS += ../shared/gExtension.h
    SOURCES += ../shared/gExtension.cpp
    INCLUDEPATH += D:/usr/local/include D:/ofl/shared \
                   D:/usr/hdf5/include \
                   D:/usr/libxml2x64/include \
                   D:/usr/eigen2
    QMAKE_LIBDIR += D:/usr/local/lib64 
    LIBS += -LD:/usr/libxml2x64/lib -llibxml2 \
            -LD:/usr/hdf5/lib -lszip
    DEFINES += OPENEXR_DLL NDEBUG NOMINMAX
    CONFIG += console
}
DESTDIR = ./
