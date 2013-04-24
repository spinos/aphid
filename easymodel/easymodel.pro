TEMPLATE = lib
CONFIG   -= qt
CONFIG   += release
PARENTPROJ = ..
SHAREDIR = ../shared
COREDIR = ../../ofl/core
OPIUMDIR = ../../ofl/opium
INCLUDEPATH = $${SHAREDDIR} \
                $${COREDIR} \
                $${OPIUMDIR} \
                $(HOME)/Library/boost_1_44_0
LIBS += -L$(HOME)/Library/boost_1_44_0/stage/lib -lboost_filesystem
LIBS += -framework libxml
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
