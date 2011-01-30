SOURCES   = main.cpp
DESTDIR        = ./
CONFIG+=console
win32 {
     INCLUDEPATH += C:\\Python26x64\\include
     QMAKE_LIBDIR += C:\\Python26x64\\libs
}
macx {
     INCLUDEPATH += /Library/Frameworks/Python.framework/Versions/2.6/include/python2.6
     QMAKE_LIBDIR += /Library/Frameworks/Python.framework/Versions/2.6/libs
    LIBS += -framework Python
}
