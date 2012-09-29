INCLUDEPATH += ../shared

HEADERS       = ../shared/Vector3F.h \
                ../shared/Matrix44F.h \
                ../shared/BaseCamera.h \
                ../shared/shapeDrawer.h \
                ../shared/Polytode.h \
                ../shared/Vertex.h \
                ../shared/Facet.h \
                ../shared/GeoElement.h \
                ../shared/Edge.h \
                ../shared/BaseMesh.h \
                ../shared/HemisphereMesh.h \
                glwidget.h \
                window.h

SOURCES       = ../shared/Vector3F.cpp \
                ../shared/Matrix44F.cpp \
                ../shared/BaseCamera.cpp \
                ../shared/shapeDrawer.cpp \
                ../shared/Polytode.cpp \
                ../shared/Vertex.cpp \
                ../shared/Facet.cpp \
                ../shared/GeoElement.cpp \
                ../shared/Edge.cpp \
                ../shared/BaseMesh.cpp \
                ../shared/HemisphereMesh.cpp \
                glwidget.cpp \
                main.cpp \
                window.cpp

QT           += opengl
win32 {
CONFIG += console
}
