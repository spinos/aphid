mac:LIBS += -leasymodel
INCLUDEPATH += ../shared 
win32:INCLUDEPATH += D:/ofl/shared D:/usr/libxml2x64/include D:/usr/eigen2
mac:INCLUDEPATH += /Users/jianzhang/Library/eigen2
win32:LIBS +=  -LD:/usr/local/lib64 -leasymodel -LD:/usr/libxml2x64/lib -llibxml2
HEADERS       = glwidget.h \
                window.h \
                ../shared/Vector3F.h \
                ../shared/Matrix44F.h \
                ../shared/BaseCamera.h \
                ../shared/Base3DView.h \
                ../shared/TypedEntity.h \
                ../shared/Geometry.h \
                ../shared/BaseMesh.h \
                ../shared/BoundingBox.h \
                ../shared/TriangleMesh.h \
                ../shared/BaseDrawer.h \
                ../shared/GeoElement.h \
                ../shared/Vertex.h \
                ../shared/Edge.h \
                ../shared/Facet.h \
                ../shared/BaseDeformer.h \
                MeshLaplacian.h \
                VertexAdjacency.h \
                LaplaceDeformer.h
                
SOURCES       = glwidget.cpp \
                main.cpp \
                window.cpp \
                ../shared/Vector3F.cpp \
                ../shared/Matrix44F.cpp \
                ../shared/BaseCamera.cpp \
                ../shared/Base3DView.cpp \
                ../shared/TypedEntity.cpp \
                ../shared/Geometry.cpp \
                ../shared/BoundingBox.cpp \
                ../shared/BaseMesh.cpp \
                ../shared/TriangleMesh.cpp \
                ../shared/BaseDrawer.cpp \
                ../shared/GeoElement.cpp \
                ../shared/Vertex.cpp \
                ../shared/Edge.cpp \
                ../shared/Facet.cpp \
                ../shared/BaseDeformer.cpp \
                MeshLaplacian.cpp \
                VertexAdjacency.cpp \
                LaplaceDeformer.cpp

QT           += opengl
win32:CONFIG += console
win32:    DEFINES += NOMINMAX
#mac:CONFIG -= app_bundle
