TEMPLATE = lib
TARGET = driver_foo
CONFIG += dll thread release
CONFIG -= qt
SOURCES = D:/usr/cortex-7.7.2/src/IECore/DisplayDriver.cpp \
            D:/usr/cortex-7.7.2/src/IECore/SimpleTypedData.cpp \
            D:/usr/cortex-7.7.2/src/IECore/MessageHandler.cpp \
            D:/usr/cortex-7.7.2/contrib/IECoreArnold/src/IECoreArnold/ToArnoldConverter.cpp \
            OutputDriver.cpp
INCLUDEPATH += D:/usr/arnoldSDK/arnold4014/include \
                D:/usr/cortex-7.7.2/include \
                D:/usr/cortex-7.7.2/contrib/IECoreArnold/include \
                D:/usr/local/include
LIBS += -LD:/usr/arnoldSDK/arnold4014/lib -lai \
        D:/usr/tbb/lib/intel64/vc9 -ltbb
