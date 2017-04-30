INCLUDEPATH += ../../shared ../ ./ ../../lapl
LIBS += -L../../lib -laphid		
CONFIG += release
HEADERS       = ../Types.h ../Entity.h ../List.h ../Sequence.h ../BNode.h ../C3Tree.h glwidget.h window.h ../../shared/Base3DView.h ../../shared/RayMarch.h \
                  ../Sculptor.h \
                  ../Dropoff.h \
				  ../Ordered.h
SOURCES       = ../Types.cpp ../Entity.cpp ../List.cpp ../Sequence.cpp ../BNode.cpp ../C3Tree.cpp glwidget.cpp window.cpp main.cpp ../../shared/Base3DView.cpp ../../shared/RayMarch.cpp \
                    ../../shared/SimpleTopology.cpp ../Sculptor.cpp ../ActiveGroup.cpp \
                    ../Dropoff.cpp \
					../Ordered.cpp
QT           += opengl
win32 {
    SOURCES += ../../shared/gExtension.cpp
    INCLUDEPATH += D:/usr/boost_1_51_0 
    QMAKE_LIBDIR += D:/usr/boost_1_51_0/stage/lib 
    CONFIG += console
}
macx {
    INCLUDEPATH += ../../../Library/boost_1_55_0
	LIBS += -lboost_date_time -lboost_system -lboost_thread -lboost_filesystem
}
