TARGET = testfem
CONFIG += release
INCLUDEPATH += ../shared ../ ./
LIBS += -L../lib -laphid -lIlmImf -lHalf
HEADERS       = ../shared/Base3DView.h \
                ../shared/BaseSolverThread.h \
                glwidget.h \
                window.h \
                 FEMTetrahedronMesh.h \
                    ConjugateGradientSolver.h \
                 SolverThread.h
SOURCES       = ../shared/Base3DView.cpp \
                ../shared/BaseSolverThread.cpp \
                glwidget.cpp \
                main.cpp \
                window.cpp \
                FEMTetrahedronMesh.cpp \
                ConjugateGradientSolver.cpp \
                SolverThread.cpp
macx {
    INCLUDEPATH += $(HOME)/Library/boost_1_55_0 \
        /usr/local/include/bullet
    LIBS += -lboost_date_time -lboost_system -lboost_thread -lboost_filesystem 
    CONFIG -= app_bundle
}
win32 {
    HEADERS += ../shared/gExtension.h
    SOURCES += ../shared/gExtension.cpp
    INCLUDEPATH += D:/usr/boost_1_51_0 /usr/OpenEXR/include
    LIBS += -LD:/usr/openEXR/lib -LD:/usr/boost_1_51_0/stage/lib
    CONFIG += console
}
QT           += opengl
DESTDIR = ./
OBJECTS_DIR = release/obj
MOC_DIR = release/moc
RCC_DIR = release/rcc
UI_DIR = release/ui
