#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>
#include <Base3DView.h>
#include "Scene.h"

namespace ttg {

class hegWidget : public aphid::Base3DView
{
    Q_OBJECT

public:
    
    hegWidget(QWidget *parent = 0);
    ~hegWidget();
	
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

};

}
#endif
