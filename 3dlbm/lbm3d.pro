INCLUDEPATH += ../shared

HEADERS       = ../shared/Vector3F.h \
                ../shared/Matrix44F.h \
                ../shared/BaseCamera.h \
                ../shared/shapeDrawer.h \
                glwidget.h \
                window.h \
                FluidContainer.h
SOURCES       = ../shared/Vector3F.cpp \
                ../shared/Matrix44F.cpp \
                ../shared/BaseCamera.cpp \
                ../shared/shapeDrawer.cpp \
                glwidget.cpp \
                main.cpp \
                window.cpp \
                FluidContainer.cpp
QT           += opengl
win32 {
CONFIG += console
}
