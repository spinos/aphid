INCLUDEPATH += ../mallard ../shared ../lapl ../catmullclark ../easymodel ../../ofl/opium ../fit
win32:INCLUDEPATH += D:/ofl/shared D:/usr/eigen3
mac:INCLUDEPATH += /Users/jianzhang/Library/eigen3
win32:LIBS +=  -LD:/usr/local/lib64
HEADERS       = ../shared/Base3DView.h \
                ../shared/QDoubleEditSlider.h \
                glwidget.h \
                window.h
                
SOURCES       = ../shared/Base3DView.cpp \
                ../shared/QDoubleEditSlider.cpp \
                glwidget.cpp \
                main.cpp \
                window.cpp

win32 {
    HEADERS += ../shared/gExtension.h
    SOURCES += ../shared/gExtension.cpp
    INCLUDEPATH += D:/usr/local/include
    QMAKE_LIBDIR += D:/usr/local/lib64
CONFIG += console
}
macx {
    INCLUDEPATH += ../../Library/boost_1_44_0
        LIBS += -lboost_date_time\
            -lboost_thread
}
QT           += opengl
win32:CONFIG += console
win32:    DEFINES += NOMINMAX
#mac:CONFIG -= app_bundle
macx {
    INCLUDEPATH += ../../Library/boost_1_44_0 \
                    ../../Library/hdf5/include \
                    /usr/local/include/OpenEXR
    QMAKE_LIBDIR += ../../Library/boost_1_44_0/stage/lib \
                    ../../Library/hdf5/lib \
                    ../easymodel \
                    ../lib
    LIBS += -lboost_date_time -lboost_thread -lboost_filesystem -lboost_system -framework libxml -laphid \
            -leasymodel -lIlmImf -lHalf -lhdf5 -lhdf5_hl
}
