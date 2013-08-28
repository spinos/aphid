TEMPLATE = lib
CONFIG   -= qt
CONFIG   += staticlib thread release
PARENTPROJ = ..
SHAREDIR = ../shared
COREDIR = ../../ofl/core
OPIUMDIR = ../../ofl/opium
INCLUDEPATH += $${SHAREDDIR} \
                $${COREDIR} \
                $${OPIUMDIR} \
                $(HOME)/Library/boost_1_44_0
                
win32:INCLUDEPATH += D:/usr/local/include \
                    D:/usr/libxml2x64/include
macx:LIBS += -L$(HOME)/Library/boost_1_44_0/stage/lib -lboost_filesystem
win32: LIBS += -LD:/usr/local/lib64 -LD:/usr/libxml2x64/lib -llibxml2
HEADERS = EasyModelIn.h \
            EasyModelOut.h \
            $${COREDIR}/zXMLDoc.h \
            $${OPIUMDIR}/animIO.h \
            $${OPIUMDIR}/transformIO.h \
            $${OPIUMDIR}/modelIO.h \
            $${SHAREDIR}/SHelper.h
SOURCES = EasyModelIn.cpp \
            EasyModelOut.cpp \
            $${COREDIR}/zXMLDoc.cpp \
            $${OPIUMDIR}/animIO.cpp \
            $${OPIUMDIR}/transformIO.cpp \
            $${OPIUMDIR}/modelIO.cpp \
            $${SHAREDIR}/SHelper.cpp
TARGET  = easymodel
