mac:INCLUDEPATH += /Users/jianzhang/Library/bullet-2.78/src/
mac:LIBS += -lBulletDynamics -L/Users/jianzhang/dyn/build/src/BulletDynamics/Release \
        -lBulletCollision -L/Users/jianzhang/dyn/build/src/BulletCollision/Release \
        -lLinearMath -L/Users/jianzhang/dyn/build/src/LinearMath/Release \
        -lBulletSoftBody \
        -leasymodel
        
win32:INCLUDEPATH += D:/usr/bullet-2.81/src
win32:LIBS += -lBulletDynamics_vs2008_x64_release -lBulletCollision_vs2008_x64_release -lBulletSoftBody_vs2008_x64_release -lLinearMath_vs2008_x64_release\ 
                -LD:/usr/bullet-2.81/lib
INCLUDEPATH += ../shared ../catmullclark
HEADERS       = glwidget.h \
                window.h \
                dynamicsSolver.h \
                ../shared/Vector3F.h \
                ../shared/Matrix44F.h \
                ../shared/BaseCamera.h \
                ../shared/Base3DView.h \
                ../catmullclark/modelIn.h \
                Muscle.h \
                MuscleFascicle.h \
                shapeDrawer.h \
                Skin.h
SOURCES       = glwidget.cpp \
                main.cpp \
                window.cpp \
                dynamicsSolver.cpp \
                ../shared/Vector3F.cpp \
                ../shared/Matrix44F.cpp \
                ../shared/BaseCamera.cpp \
                ../shared/Base3DView.cpp \
                Muscle.cpp \
                MuscleFascicle.cpp \
                shapeDrawer.cpp \
                Skin.cpp
QT           += opengl
win32:CONFIG += console
##mac:CONFIG -= app_bundle
