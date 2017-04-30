#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>
#include <Base3DView.h>
#include "Scene.h"

namespace ttg {

class vdfWidget : public aphid::Base3DView
{
    Q_OBJECT

public:
    
    vdfWidget(ttg::Scene * sc, QWidget *parent = 0);
    ~vdfWidget();
	
protected:    
    virtual void clientInit();
    virtual void clientDraw();
    virtual void clientSelect(aphid::Vector3F & origin, aphid::Vector3F & ray, aphid::Vector3F & hit);
    virtual void clientDeselect();
    virtual void clientMouseInput(aphid::Vector3F & stir);
	virtual void keyPressEvent(QKeyEvent *event);
	virtual void keyReleaseEvent(QKeyEvent *event);
    
public slots:
		
private:

private slots:

private:
	ttg::Scene * m_scene;
};

}
#endif
