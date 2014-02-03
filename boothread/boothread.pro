INCLUDEPATH += ../shared \
                ../../Library/boost_1_44_0 \
                D:/usr/local/include
               
SOURCES       = main.cpp
CONFIG += release
win32 {
	INCLUDEPATH +=D:/usr/boost_1_51_0
       CONFIG+=console
        DEFINES += BOOST_USE_WINDOWS_H
     QMAKE_LIBDIR += D:/usr/boost_1_51_0/stage/lib
     LIBS += -lboost_system-mgw48-mt-1_51.dll -lboost_thread-mgw48-mt-1_51.dll 
}
macx {
    CONFIG -= app_bundle
    INCLUDEPATH += ../../Library/boost_1_44_0
     QMAKE_LIBDIR += ../../Library/boost_1_44_0/stage/lib
    LIBS += -lboost_date_time\
            -lboost_thread
}
