/*
 *  TexcoordWidget.h
 *  garden
 *
 */

#ifndef GAR_TEXCOORD_WIDGET_H
#define GAR_TEXCOORD_WIDGET_H

#include <qt/Base2DView.h>

namespace aphid {
class ATriangleMesh;
}

class ShrubScene;
class GardenGlyph;
class PieceAttrib;

class TexcoordWidget : public aphid::Base2DView
{
    Q_OBJECT

public:

    TexcoordWidget(ShrubScene* scene, QWidget *parent = 0);
    ~TexcoordWidget();
	
protected:    
    virtual void clientInit();
    virtual void clientDraw();
    virtual void clientSelect(aphid::Vector3F & origin, aphid::Vector3F & ray, aphid::Vector3F & hit);
    virtual void clientDeselect();
    virtual void clientMouseInput(aphid::Vector3F & stir);
	virtual void keyPressEvent(QKeyEvent *event);
	virtual void keyReleaseEvent(QKeyEvent *event);
    virtual void resetOrthoViewTransform();
	
public slots:
	void recvSelectGlyph(bool x);
	
private:
	void drawTexcoord(const aphid::ATriangleMesh* msh);
	
private slots:

private:
	ShrubScene* m_scene;
	GardenGlyph* m_selectedGlyph;
	
};

#endif
