SOURCES   = main.cpp 
mac:INCLUDEPATH += /Users/jianzhang/Library/eigen3
win32:INCLUDEPATH += D:/usr/eigen3
CONFIG += release
CONFIG -= qt
win32:CONFIG += console
mac:CONFIG -= app_bundle
