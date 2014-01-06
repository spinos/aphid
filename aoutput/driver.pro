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
win32 {
INCLUDEPATH += D:/usr/boost_1_51_0 \
                    D:/usr/local/openEXR/include \
                    D:/ofl/shared
QMAKE_LIBDIR += D:/usr/boost_1_51_0/stage/lib \
                    D:/usr/local/openEXR/lib
DEFINES += OPENEXR_DLL NDEBUG NOMINMAX _WIN32_WINDOWS
}
