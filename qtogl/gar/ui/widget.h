/*
 *  widget.h
 *  garden
 *
 */

#ifndef GAR_GLWIDGET_H
#define GAR_GLWIDGET_H

#include <qt/Base3DView.h>

class ShrubScene;
class Vegetation;
class VegetationPatch;
class DrawVegetation;
class PieceAttrib;

namespace gar {
class SynthesisGroup;
}

class GLWidget : public aphid::Base3DView
{
    Q_OBJECT

public:

    GLWidget(Vegetation * vege, ShrubScene* scene, QWidget *parent = 0);
    ~GLWidget();
	
	void setDisplayState(int x);
	void setViewState(int x);
	
protected:    
    virtual void clientInit();
    virtual void clientDraw();
    virtual void clientSelect(aphid::Vector3F & origin, aphid::Vector3F & ray, aphid::Vector3F & hit);
    virtual void clientDeselect();
    virtual void clientMouseInput(aphid::Vector3F & stir);
	virtual void keyPressEvent(QKeyEvent *event);
	virtual void keyReleaseEvent(QKeyEvent *event);
    virtual void resetPerspViewTransform();
	virtual void resetOrthoViewTransform();
	
public slots:
	void recvToolAction(int x);
	
private:
	void drawAsset();
    void drawSynthesis();
    void simpleDraw(VegetationPatch * vgp);
	void geomDraw(VegetationPatch * vgp);
	void pointDraw(VegetationPatch * vgp);
	void dopDraw(VegetationPatch * vgp);
	void voxelDraw(VegetationPatch * vgp);
	void diffDraw(VegetationPatch * vgp);
	void drawVariableAsset(PieceAttrib* attr);
	void drawSingleAsset(PieceAttrib* attr);
	void drawTwigAsset(PieceAttrib* attr);
	void drawSynthesisGroup(PieceAttrib* attr, gar::SynthesisGroup* grp);
	void drawBranchAsset(PieceAttrib* attr);
	
private slots:

private:
	ShrubScene* m_scene;
	Vegetation * m_vege;
	DrawVegetation * m_vegd;
	int m_viewState;
	int m_dspState;
	
};

#endif
