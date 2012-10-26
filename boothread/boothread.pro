INCLUDEPATH += ../shared \
                ../../Library/boost_1_44_0 \
                D:/usr/local/include
                
QMAKE_LIBDIR += D:/usr/local/lib64

SOURCES       = main.cpp
CONFIG += release
win32 {
       CONFIG+=console
     QMAKE_LIBDIR += D:/usr/lcoal/include/lib64
}
macx {
    CONFIG -= app_bundle
    INCLUDEPATH += ../../Library/boost_1_44_0
     QMAKE_LIBDIR += ../../Library/boost_1_44_0/stage/lib
    LIBS += -lboost_date_time\
            -lboost_thread
}
