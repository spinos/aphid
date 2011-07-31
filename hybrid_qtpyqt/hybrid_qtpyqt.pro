CONFIG   += release
HEADERS = ../hoatzin/hoatzin.h
SOURCES   = main.cpp
INCLUDEPATH += /Library/Frameworks/Python.framework/Versions/2.6/include/python2.6
LIBS += -framework Python -lhoatzin
