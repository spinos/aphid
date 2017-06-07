/*
 *  widget.h
 *  hes viewer
 *
 */

#ifndef HES_GLWIDGET_H
#define HES_GLWIDGET_H

#include <qt/Base3DView.h>

namespace aphid {

class HesScene;
class ATriangleMeshGroup;

namespace cvx {
class Triangle;
}

namespace sdb {

template<typename T>
class VectorArray;

}

class ATriangleMesh;

}

class GLWidget : public aphid::Base3DView
{
    Q_OBJECT

public:

    GLWidget(const aphid::HesScene* scene, QWidget *parent = 0);
    ~GLWidget();
	
	void setDisplayState(int x);
	
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
	
	void drawMesh(const aphid::ATriangleMeshGroup* msh);
	
public slots:
	void recvToolAction(int x);
	
private:
    
private slots:

private:
	int m_dspState;
	const aphid::HesScene* m_scene;
	
};

#endif
