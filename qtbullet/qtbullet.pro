INCLUDEPATH += /Users/jianzhang/Library/bullet-2.78/src/
mac:LIBS += -lBulletDynamics -L/Users/jianzhang/dyn/build/src/BulletDynamics/Release \
        -lBulletCollision -L/Users/jianzhang/dyn/build/src/BulletCollision/Release \
        -lLinearMath -L/Users/jianzhang/dyn/build/src/LinearMath/Release

HEADERS       = glwidget.h \
                window.h \
                dynamicsSolver.h \
                shapeDrawer.h
SOURCES       = glwidget.cpp \
                main.cpp \
                window.cpp \
                dynamicsSolver.cpp \
                shapeDrawer.cpp
QT           += opengl

