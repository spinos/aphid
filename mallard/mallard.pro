win32 {
    Aphid = D:/aphid
}
mac {
    Aphid = $$(HOME)/aphid
}

INCLUDEPATH += ./ ../shared ../lapl ../catmullclark ../easymodel ../../ofl/opium
HEADERS       =	../shared/ContextIconFrame.h \
../shared/StateIconFrame.h \
                                ../shared/ActionIconFrame.h \
                                ../shared/QIconFrame.h \
                ../shared/QIntEditSlider.h \
                ../shared/QDoubleEditSlider.h \
../shared/QColorBox.h \
../shared/QColorEditSlider.h \
../shared/AllEdit.h \
../shared/QModelEdit.h \
../shared/QColorEdit.h \
../shared/QIntEdit.h \
../shared/QDoubleEdit.h \
../shared/QBoolEdit.h \
../shared/QFileNameEdit.h \
                ../shared/Base3DView.h \
                ../shared/Base2DView.h \
                ManipulateView.h \
                ToolBox.h \
BodyMaps.h \
MlScene.h \
ScenePort.h \
                glwidget.h \
				MlCalamus.h \
				MlCalamusArray.h \
				MlDrawer.h \
    MlCache.h \
				MlTessellate.h \
    CalamusSkin.h \
PaintFeather.h \
				MlSkin.h \
                MlRachis.h \
                BrushControl.h \
SelectFaceBox.h \
CombBox.h \
CurlBox.h \
ScaleBox.h \
WidthBox.h \
FloodBox.h \
EraseBox.h \
PaintBox.h \
PitchBox.h \
				HFeather.h \
    HSkin.h \
                MlFeatherCollection.h \
	HLight.h \
HShader.h \
	HOption.h \
    MlUVView.h \
    FeatherEdit.h \
	FeatherEditTool.h \
	TimeControl.h \
    LaplaceSmoother.h \
	MlCluster.h \
	BaseVane.h \
	MlVane.h \
	BarbEdit.h \
	BarbView.h \
	FeatherExample.h \
    BaseFeather.h \
	TexturedFeather.h \
    DeformableFeather.h \
    MlFeather.h \
    BarbControl.h \
	RenderEdit.h \
    ImageView.h \
	MlEngine.h \
	AnorldFunc.h \
	BarbWorks.h \
	FeatherAttrib.h \
	SceneEdit.h \
SceneTreeModel.h \
SceneTreeParser.h \
SceneTreeItem.h \
EditDelegate.h \
MlShaders.h \
FeatherShader.h \
                window.h
                
SOURCES       = ../shared/ContextIconFrame.cpp \
../shared/StateIconFrame.cpp \
				../shared/ActionIconFrame.cpp \
				../shared/QIconFrame.cpp \
				../shared/QIntEditSlider.cpp \
                ../shared/QDoubleEditSlider.cpp \
../shared/QColorBox.cpp \
../shared/QColorEditSlider.cpp \
../shared/QModelEdit.cpp \
../shared/QColorEdit.cpp \
../shared/QIntEdit.cpp \
../shared/QDoubleEdit.cpp \
../shared/QBoolEdit.cpp \
../shared/QFileNameEdit.cpp \
                ../shared/Base3DView.cpp \
                ../shared/Base2DView.cpp \
                ManipulateView.cpp \
                ToolBox.cpp \
BodyMaps.cpp \
MlScene.cpp \
ScenePort.cpp \
                glwidget.cpp \
				MlCalamus.cpp \
				MlCalamusArray.cpp \
				MlDrawer.cpp \
    MlCache.cpp \
				MlTessellate.cpp \
    CalamusSkin.cpp \
PaintFeather.cpp \
				MlSkin.cpp \
                MlRachis.cpp \
                BrushControl.cpp \
SelectFaceBox.cpp \
CombBox.cpp \
CurlBox.cpp \
ScaleBox.cpp \
WidthBox.cpp \
FloodBox.cpp \
EraseBox.cpp \
PaintBox.cpp \
PitchBox.cpp \
				HFeather.cpp \
    HSkin.cpp \
                MlFeatherCollection.cpp \
	HLight.cpp \
HShader.cpp \
	HOption.cpp \
    MlUVView.cpp \
    FeatherEdit.cpp \
	FeatherEditTool.cpp \
	TimeControl.cpp \
    LaplaceSmoother.cpp \
	MlCluster.cpp \
	BaseVane.cpp \
	MlVane.cpp \
	BarbEdit.cpp \
	BarbView.cpp \
	FeatherExample.cpp \
    BaseFeather.cpp \
	TexturedFeather.cpp \
        DeformableFeather.cpp \
        MlFeather.cpp \
    BarbControl.cpp \
	RenderEdit.cpp \
    ImageView.cpp \
	MlEngine.cpp \
	BarbWorks.cpp \
	AnorldFunc.cpp \
	FeatherAttrib.cpp \
	SceneEdit.cpp \
SceneTreeModel.cpp \
SceneTreeParser.cpp \
SceneTreeItem.cpp \
EditDelegate.cpp \
MlShaders.cpp \
FeatherShader.cpp \
    window.cpp \
                main.cpp
                
LIBS += -L../easymodel -leasymodel \
        -L../lib -laphid \
        -lIlmImf -lHalf -lhdf5 -lhdf5_hl
macx {
    INCLUDEPATH += ../../Library/boost_1_44_0 \
                    ../../Library/hdf5/include \
                    /usr/local/include/OpenEXR \
                    /Users/jianzhang/Library/eigen2
    QMAKE_LIBDIR += ../../Library/boost_1_44_0/stage/lib \
                    ../../Library/hdf5/lib
    LIBS += -lboost_date_time -lboost_thread -lboost_filesystem -lboost_system -framework libxml
}
win32 {
    HEADERS += ../shared/gExtension.h
    SOURCES += ../shared/gExtension.cpp
    INCLUDEPATH += D:/usr/boost_1_51_0 \
                    D:/usr/local/openEXR/include \
                    D:/ofl/shared \
                   D:/usr/hdf5/include \
                   D:/usr/libxml2x64/include \
                   D:/usr/eigen2 \
                   D:/usr/arnoldSDK/arnold4014/include
    QMAKE_LIBDIR += D:/usr/boost_1_51_0/stage/lib \
                    D:/usr/local/openEXR/lib
    LIBS += -LD:/usr/libxml2x64/lib -llibxml2 \
            -LD:/usr/hdf5/lib -lszip \
            -LD:/usr/arnoldSDK/arnold4014/lib -lai
    DEFINES += OPENEXR_DLL NDEBUG NOMINMAX _WIN32_WINDOWS
    CONFIG += console
}
QT           += opengl
RESOURCES += ../icons/mallard.qrc
DESTDIR = ./
