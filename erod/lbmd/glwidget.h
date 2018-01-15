/*
 *  lbm collision
 */
#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <qt/Base3DView.h>

namespace aphid {

namespace lbm {

class LatticeManager;

}

}

class GLWidget : public aphid::Base3DView
{
    Q_OBJECT

public:

    GLWidget(QWidget *parent = 0);
    ~GLWidget();
	
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
	
private:
	
private:
    aphid::lbm::LatticeManager* m_latman;
	
};

#endif
