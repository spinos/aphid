SOURCES   = main.cpp 
mac:INCLUDEPATH += /Users/jianzhang/Library/eigen2
win32:INCLUDEPATH += D:/usr/eigen2
CONFIG += release
CONFIG -= qt
win32:CONFIG += console
mac:CONFIG -= app_bundle
