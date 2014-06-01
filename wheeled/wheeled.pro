INCLUDEPATH += ../shared ../ ./ ../caterpillar
LIBS += -L../lib -laphid -lIlmImf -lHalf
macx {
    INCLUDEPATH += $(HOME)/Library/boost_1_55_0 $(HOME)/Library/bullet-2.81/src/ \
        /usr/local/include/bullet
    LIBS += -lboost_date_time -lboost_system -lboost_thread -lboost_filesystem \
        -L/usr/local/lib -lBulletDynamics -lBulletCollision -lBulletSoftBody -lLinearMath

}
win32 {
    INCLUDEPATH += D:/usr/bullet-2.82/src \
                    D:/usr/boost_1_51_0 \
                    D:/usr/local/openEXR/include
    QMAKE_LIBDIR += D:/usr/boost_1_51_0/stage/lib \
                    D:/usr/local/openEXR/lib \
                    D:/usr/bullet-2.82/lib
    LIBS += -lBulletDynamics -lBulletCollision -lBulletSoftBody -lLinearMath 
    QMAKE_LFLAGS_RELEASE += /NODEFAULTLIB:libcmt  /NODEFAULTLIB:libcpmt
}

HEADERS       = glwidget.h \
                window.h \
                ../tracked/DynamicsSolver.h \
                ../shared/Base3DView.h \
                ../caterpillar/GroupId.h \
                ../caterpillar/PhysicsState.h \
				../caterpillar/Common.h \
                ../tracked/shapeDrawer.h \
				MeshShape.h \
TireMesh.h \
				Suspension.h\
				Wheel.h \
WheeledChassis.h \
				WheeledVehicle.h
SOURCES       = glwidget.cpp \
                main.cpp \
                window.cpp \
                ../tracked/DynamicsSolver.cpp \
                ../shared/Base3DView.cpp \
                ../caterpillar/GroupId.cpp \
                ../caterpillar/PhysicsState.cpp \
                ../tracked/shapeDrawer.cpp \
				MeshShape.cpp \
TireMesh.cpp \
				Suspension.cpp \
				Wheel.cpp \
WheeledChassis.cpp \
				WheeledVehicle.cpp
QT           += opengl
win32:CONFIG += console
##mac:CONFIG -= app_bundle
