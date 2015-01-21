INCLUDEPATH += ../../shared ../ ./
LIBS += -L../../lib -laphid -lIlmImf -lHalf
DESTDIR = ./
macx {
    INCLUDEPATH += $(HOME)/Library/boost_1_55_0 \
        /usr/local/include/bullet
    LIBS += -lboost_date_time -lboost_system -lboost_thread -lboost_filesystem 
}
win32 {
    HEADERS += ../../shared/gExtension.h
    SOURCES += ../../shared/gExtension.cpp
    INCLUDEPATH += D:/usr/boost_1_51_0 \
                    D:/usr/local/openEXR/include
    QMAKE_LIBDIR += D:/usr/boost_1_51_0/stage/lib \
                    D:/usr/local/openEXR/lib
    QMAKE_LFLAGS_RELEASE += /NODEFAULTLIB:libcmt  /NODEFAULTLIB:libcpmt
}

HEADERS       = glwidget.h \
                window.h \
                ../../shared/Base3DView.h \
                ../Gjk.h \
				../GjkContactSolver.h
SOURCES       = ../../shared/Base3DView.cpp \
				glwidget.cpp \
                main.cpp \
                window.cpp \
                ../Gjk.cpp \
				../GjkContactSolver.cpp 
QT           += opengl
win32:CONFIG += console
macx:CONFIG -= app_bundle 
