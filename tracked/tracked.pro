INCLUDEPATH += ../shared ../ ./
LIBS += -L../lib -laphid -lIlmImf -lHalf
macx {
    INCLUDEPATH += $(HOME)/Library/boost_1_55_0 $(HOME)/Library/bullet-2.81/src/ \
        /usr/local/include/bullet
    LIBS += -lboost_date_time -lboost_system -lboost_thread -lboost_filesystem \
        -L/usr/local/lib -lBulletDynamics -lBulletCollision -lBulletSoftBody -lLinearMath

}
win32:INCLUDEPATH += D:/usr/bullet-2.81/src D:/ofl/shared D:/usr/libxml2x64/include
win32:LIBS += -lBulletDynamics_vs2008_x64_release -lBulletCollision_vs2008_x64_release -lBulletSoftBody_vs2008_x64_release -lLinearMath_vs2008_x64_release\ 
                -LD:/usr/bullet-2.81/lib \
                -LD:/usr/local/lib64 -leasymodel -LD:/usr/libxml2x64/lib -llibxml2
HEADERS       = glwidget.h \
                window.h \
                DynamicsSolver.h \
                ../shared/Base3DView.h \
                shapeDrawer.h \
				Tread.h \
                TrackedPhysics.h
SOURCES       = glwidget.cpp \
                main.cpp \
                window.cpp \
                DynamicsSolver.cpp \
                ../shared/Base3DView.cpp \
                shapeDrawer.cpp \
				Tread.cpp \
                TrackedPhysics.cpp
QT           += opengl
win32:CONFIG += console
##mac:CONFIG -= app_bundle
