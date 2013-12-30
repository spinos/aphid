TEMPLATE = lib
TARGET = driver_foo
CONFIG += dll thread release
CONFIG -= qt
SOURCES = OutputDriver.cpp
INCLUDEPATH += D:/usr/arnoldSDK/arnold4014/include
QMAKE_LIBDIR += D:/usr/local/lib64 
LIBS += -LD:/usr/arnoldSDK/arnold4014/lib -lai \
        -LD:/usr/tbb/lib/intel64/vc9 -ltbb
DESTDIR = ./
