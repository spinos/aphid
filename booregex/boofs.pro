SOURCES   = main.cpp
DESTDIR        = ./
CONFIG+=console
win32 {
     INCLUDEPATH += C:\\Python26x64\\include
     QMAKE_LIBDIR += C:\\Python26x64\\libs
}
macx {
     INCLUDEPATH += /Users/jianzhang/Library/boost_1_44_0
     QMAKE_LIBDIR += /Users/jianzhang/Library/boost_1_44_0/stage/lib
    LIBS += -lboost_filesystem\
            -lboost_system\
            -lboost_regex
}
