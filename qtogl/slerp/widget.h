/* 
 *  slerp test
 */
 
#ifndef HERMITE_WIDGET_H
#define HERMITE_WIDGET_H

#include <qt/Base3DView.h>

namespace aphid {

class Vector3F;
class Matrix44F;
class TranslationHandle;
class RotationHandle;
class Ray;
class Fiber;
class FiberBundle;
class FiberBundleBuilder;
class DrawArrow;

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
	virtual void clientSelect(QMouseEvent *event);
    virtual void clientDeselect(QMouseEvent *event);
    virtual void clientMouseInput(QMouseEvent *event);
	virtual void keyPressEvent(QKeyEvent *event);
	
public slots:
	void recvToolSelected(int x);
	
signals:

private:
	void drawPoints();
	void selectPoint(const aphid::Ray* incident);
	void movePoint();
	void rotatePoint();
	void moveStrand();
	void rotateStrand();
	void drawInterp();
	void drawFiberInterp(const aphid::Fiber* fib,
				aphid::DrawArrow* darr);
	void drawBuilder();
	
private:
	aphid::TranslationHandle * m_tranh;
	aphid::RotationHandle * m_roth;
	aphid::Matrix44F* m_space;
	aphid::Ray* m_incident;
	aphid::FiberBundleBuilder* m_fbbld;
	aphid::FiberBundle* m_fib;
	int m_mode;
	
};

#endif
