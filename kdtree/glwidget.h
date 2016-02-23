#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <Base3DView.h>
#include "SceneContainer.h"

class GLWidget : public Base3DView
{
    Q_OBJECT

public:
    GLWidget(QWidget *parent = 0);
    ~GLWidget();
public slots:
	
signals:

protected:
    virtual void clientInit();
    virtual void clientDraw();
	virtual void clientSelect(QMouseEvent *event);
    virtual void clientDeselect(QMouseEvent *event);
    virtual void clientMouseInput(QMouseEvent *event);
	virtual void keyPressEvent(QKeyEvent *event);
	
private:	
	SceneContainer* m_scene;
private slots:
    

};
#endif
