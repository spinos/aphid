INCLUDEPATH += ../shared

HEADERS       = ../shared/BaseArray.h \
                ../shared/TypedEntity.h \
                ../shared/Primitive.h
SOURCES       = ../shared/BaseArray.cpp \
                ../shared/TypedEntity.cpp \
                ../shared/Primitive.cpp \
                main.cpp
win32 {
CONFIG += console
}
mac {
  CONFIG -= app_bundle
}

