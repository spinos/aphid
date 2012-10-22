INCLUDEPATH += ../shared

HEADERS       = ../shared/BaseArray.h
SOURCES       = ../shared/BaseArray.cpp \
                main.cpp
win32 {
CONFIG += console
}
mac {
  CONFIG -= app_bundle
}

