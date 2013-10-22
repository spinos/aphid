win32 {
    Aphid = D:/aphid
}
mac {
    Aphid = $$(HOME)/aphid
}

INCLUDEPATH += ./ ../shared ../lapl ../catmullclark ../easymodel ../../ofl/opium ../fit
HEADERS       =	../shared/ContextIconFrame.h \
                                ../shared/ActionIconFrame.h \
                                ../shared/QIconFrame.h \
                ../shared/QIntEditSlider.h \
                ../shared/QDoubleEditSlider.h \
                ../shared/Base3DView.h \
                ../shared/Base2DView.h \
                ../fit/SingleModelView.h \
                ToolBox.h \
                glwidget.h \
                MlFeather.h \
				MlCalamus.h \
				MlCalamusArray.h \
				MlDrawer.h \
    MlCache.h \
				MlTessellate.h \
				MlSkin.h \
                MlRachis.h \
                BrushControl.h \
                MlScene.h \
				HFeather.h \
    HSkin.h \
                MlFeatherCollection.h \
    MlUVView.h \
    FeatherEdit.h \
	FeatherEditTool.h \
	TimeControl.h \
                window.h
                
SOURCES       = ../shared/ContextIconFrame.cpp \
				../shared/ActionIconFrame.cpp \
				../shared/QIconFrame.cpp \
				../shared/QIntEditSlider.cpp \
                ../shared/QDoubleEditSlider.cpp \
                ../shared/Base3DView.cpp \
                ../shared/Base2DView.cpp \
                ../fit/SingleModelView.cpp \
                ToolBox.cpp \
                glwidget.cpp \
                MlFeather.cpp \
				MlCalamus.cpp \
				MlCalamusArray.cpp \
				MlDrawer.cpp \
    MlCache.cpp \
				MlTessellate.cpp \
				MlSkin.cpp \
                MlRachis.cpp \
                BrushControl.cpp \
                MlScene.cpp \
				HFeather.cpp \
    HSkin.cpp \
                MLFeatherCollection.cpp \
    MlUVView.cpp \
    FeatherEdit.cpp \
	FeatherEditTool.cpp \
	TimeControl.cpp \
    window.cpp \
                main.cpp
                
LIBS += -L../lib -laphid -L../easymodel -leasymodel -lIlmImf -lHalf -lhdf5 -lhdf5_hl
macx {
    INCLUDEPATH += ../../Library/boost_1_44_0 \
                    ../../Library/hdf5/include \
                    /usr/local/include/OpenEXR
    QMAKE_LIBDIR += ../../Library/boost_1_44_0/stage/lib \
                    ../../Library/hdf5/lib
    LIBS += -lboost_date_time -lboost_thread -lboost_filesystem -lboost_system -framework libxml
}
win32 {
    HEADERS += ../shared/gExtension.h
    SOURCES += ../shared/gExtension.cpp
    INCLUDEPATH += D:/usr/local/include D:/ofl/shared \
                   D:/usr/hdf5/include \
                   D:/usr/libxml2x64/include D:/usr/eigen3
    QMAKE_LIBDIR += D:/usr/local/lib64 
    LIBS += -LD:/usr/libxml2x64/lib -llibxml2 \
            -LD:/usr/hdf5/lib -lszip
    DEFINES += OPENEXR_DLL NDEBUG
    CONFIG += console
}
QT           += opengl
RESOURCES += ../icons/mallard.qrc
