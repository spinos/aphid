mac:INCLUDEPATH += /Users/jianzhang/Library/bullet-2.78/src/
mac:LIBS += -lBulletDynamics -L/Users/jianzhang/dyn/build/src/BulletDynamics/Release \
        -lBulletCollision -L/Users/jianzhang/dyn/build/src/BulletCollision/Release \
        -lLinearMath -L/Users/jianzhang/dyn/build/src/LinearMath/Release \
        -lBulletSoftBody
INCLUDEPATH += ../shared ../ ./
LIBS += -L../lib -laphid -lIlmImf -lHalf
macx {
    INCLUDEPATH += $(HOME)/Library/boost_1_44_0
    LIBS += -lboost_date_time -lboost_system -lboost_thread -lboost_filesystem
}
win32:INCLUDEPATH += D:/usr/bullet-2.81/src D:/ofl/shared D:/usr/libxml2x64/include
win32:LIBS += -lBulletDynamics_vs2008_x64_release -lBulletCollision_vs2008_x64_release -lBulletSoftBody_vs2008_x64_release -lLinearMath_vs2008_x64_release\ 
                -LD:/usr/bullet-2.81/lib \
                -LD:/usr/local/lib64 -leasymodel -LD:/usr/libxml2x64/lib -llibxml2
HEADERS       = glwidget.h \
                window.h \
                dynamicsSolver.h \
                ../shared/Base3DView.h \
                shapeDrawer.h
SOURCES       = glwidget.cpp \
                main.cpp \
                window.cpp \
                dynamicsSolver.cpp \
                ../shared/Base3DView.cpp \
                shapeDrawer.cpp
QT           += opengl
win32:CONFIG += console
##mac:CONFIG -= app_bundle
