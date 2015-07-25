#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>
#include <Base3DView.h>
class LarixWorld;
class GLWidget : public Base3DView
{
    Q_OBJECT

public:
    
    GLWidget(QWidget *parent = 0);
    ~GLWidget();
	
protected:  
	virtual void clientInit();
    virtual void clientDraw();
    virtual void clientSelect(QMouseEvent *event);
    virtual void clientDeselect(QMouseEvent *event);
    virtual void clientMouseInput(QMouseEvent *event);
	virtual void keyPressEvent(QKeyEvent *event);
    
private:
	LarixWorld * m_world;

private slots:
    

};
#endif
