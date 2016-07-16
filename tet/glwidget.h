#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>
#include <Base3DView.h>
#include "Scene.h"

namespace ttg {

class GLWidget : public aphid::Base3DView
{
    Q_OBJECT

public:
    
    GLWidget(ttg::Scene * sc, QWidget *parent = 0);
    ~GLWidget();
	
protected:    
    virtual void clientInit();
    virtual void clientDraw();
    virtual void clientSelect(aphid::Vector3F & origin, aphid::Vector3F & ray, aphid::Vector3F & hit);
    virtual void clientDeselect();
    virtual void clientMouseInput(aphid::Vector3F & stir);
	virtual void keyPressEvent(QKeyEvent *event);
	virtual void keyReleaseEvent(QKeyEvent *event);
    
public slots:
	void receiveA1(double x);
	void receiveB1(double x);
	void receiveM1(double x);
	void receiveN1(double x);
	void receiveN2(double x);
	void receiveN3(double x);

	void receiveA2(double x);
	void receiveB2(double x);
	void receiveM2(double x);
	void receiveN21(double x);
	void receiveN22(double x);
	void receiveN23(double x);
	
	void receiveA(double x);
	void receiveB(double x);
	void receiveC(double x);
	void receiveD(double x);
	
private:

private slots:

private:
	ttg::Scene * m_scene;
};

}
#endif
