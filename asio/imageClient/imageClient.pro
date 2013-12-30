win32 {
    Aphid = D:/aphid
}
mac {
    Aphid = $$(HOME)/aphid
}
CONFIG -= qt
INCLUDEPATH += ./ ../shared ../lapl ../catmullclark ../easymodel ../../ofl/opium ../fit
SOURCES       = main.cpp
                
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
    INCLUDEPATH += D:/usr/boost_1_51_0 \
                    D:/ofl/shared 
    QMAKE_LIBDIR += D:/usr/boost_1_51_0/stage/lib
    LIBS += -LD:/usr/hdf5/lib -lszip
    DEFINES += NDEBUG NOMINMAX _WIN32_WINDOWS
    CONFIG += console
}
DESTDIR = ./
