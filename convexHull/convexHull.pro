INCLUDEPATH += ../shared

HEADERS       = ../shared/Vector3F.h \
                ../shared/Matrix44F.h \
                ../shared/BaseCamera.h \
                ../shared/shapeDrawer.h \
                glwidget.h \
                window.h \
                HullContainer.h \
                ../shared/Vertex.h \
                ../shared/Facet.h \
                ../shared/GeoElement.h \
                ../shared/GraphArch.h \
                ../shared/ConflictGraph.h \
                ../shared/Edge.h
SOURCES       = ../shared/Vector3F.cpp \
                ../shared/Matrix44F.cpp \
                ../shared/BaseCamera.cpp \
                ../shared/shapeDrawer.cpp \
                glwidget.cpp \
                main.cpp \
                window.cpp \
                HullContainer.cpp \
                ../shared/Vertex.cpp \
                ../shared/Facet.cpp \
                ../shared/GeoElement.cpp \
                ../shared/GraphArch.cpp \
                ../shared/ConflictGraph.cpp \
                ../shared/Edge.cpp
QT           += opengl
win32 {
CONFIG += console
}
