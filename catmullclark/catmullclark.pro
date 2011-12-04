HEADERS       = accPatch.h \
                accStencil.h \
                bezierPatch.h \
                patchTopology.h \
                tessellator.h \
                glwidget.h \
                subdivision.h \
                window.h \
                modelIn.h
SOURCES       = accPatch.cpp \
                accStencil.cpp \
                bezierPatch.cpp \
                patchTopology.cpp \
                tessellator.cpp \
                glwidget.cpp \
                subdivision.cpp \
                main.cpp \
                window.cpp
INCLUDEPATH = /usr/local/include/OpenEXR
LIBS          += -leasymodel
QT           += opengl

