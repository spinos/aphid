SOURCES   = main.cpp
DESTDIR        = ./
win32 {
       CONFIG+=console
     INCLUDEPATH += C:\\Python26x64\\include
     QMAKE_LIBDIR += C:\\Python26x64\\libs
}
macx {
    CONFIG -= app_bundle
     INCLUDEPATH += /Users/jianzhang/Library/boost_1_44_0
     QMAKE_LIBDIR += /Users/jianzhang/Library/boost_1_44_0/stage/lib
    LIBS += -lboost_filesystem\
            -lboost_system\
            -lboost_regex
}
