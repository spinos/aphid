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
                ../shared/BaseBuffer.h \
                glwidget.h \
                window.h \
                SceneContainer.h \
                RandomMesh.h
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
                ../shared/BaseBuffer.cpp \
                glwidget.cpp \
                main.cpp \
                window.cpp \
                SceneContainer.cpp \
                RandomMesh.cpp
QT           += opengl
win32 {
CONFIG += console
}
