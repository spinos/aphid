INCLUDEPATH += ../mallard ../shared ../lapl ../catmullclark ../easymodel ../../ofl/opium ../fit
win32:INCLUDEPATH += D:/ofl/shared D:/usr/eigen3
mac:INCLUDEPATH += /Users/jianzhang/Library/eigen3
LIBS += -L../easymodel -leasymodel -lIlmImf -lHalf -lhdf5 -lhdf5_hl -L../lib -laphid
HEADERS       = ../shared/Base3DView.h \
                ../shared/QDoubleEditSlider.h \
                ../shared/QDouble3Edit.h \
                glwidget.h \
                SkeletonJointEdit.h \
                window.h
                
SOURCES       = ../shared/Base3DView.cpp \
                ../shared/QDoubleEditSlider.cpp \
                ../shared/QDouble3Edit.cpp \
                glwidget.cpp \
                main.cpp \
                SkeletonJointEdit.cpp \
                window.cpp

win32 {
    HEADERS += ../shared/gExtension.h
    SOURCES += ../shared/gExtension.cpp
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
win32 {
    INCLUDEPATH += D:/usr/local/include
    QMAKE_LIBDIR += D:/usr/local/lib64
    LIBS += -LD:/usr/libxml2x64/lib -llibxml2 \
            -LD:/usr/hdf5/lib -lszip          
    DEFINES += OPENEXR_DLL NDEBUG
    CONFIG += console
}
