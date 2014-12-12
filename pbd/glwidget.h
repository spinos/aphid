#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>
#include <Base3DView.h>

class GLWidget : public Base3DView
{
    Q_OBJECT

public:
	struct Spring {
		unsigned p1, p2;
		float rest_length;
		float Ks, Kd;
		int type;
	};
    GLWidget(QWidget *parent = 0);
    ~GLWidget();
	
protected:
	virtual void clientInit();
    virtual void clientDraw();
    virtual void clientSelect(Vector3F & origin, Vector3F & ray, Vector3F & hit);
    virtual void clientDeselect();
    virtual void clientMouseInput(Vector3F & stir);
	virtual void keyPressEvent(QKeyEvent *event);
	virtual void keyReleaseEvent(QKeyEvent *event);
    
//! [3]
private:
	Vector3F * m_pos;
	Vector3F * m_posLast;
	Vector3F * m_force;
	unsigned * m_indices;
	Spring * m_spring;
	unsigned m_numSpring;
	void setSpring(Spring * dest, unsigned a, unsigned b, float ks, float kd, int type);
	void stepPhysics(float dt);
	void computeForces(float dt);
	void integrateVerlet(float dt);
	static Vector3F getVerletVelocity(Vector3F x_i, Vector3F xi_last, float dt );
private slots:
    void simulate();

};
//! [3]

#endif
