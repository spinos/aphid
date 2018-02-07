#ifndef SDFT_WIDGET_H
#define SDFT_WIDGET_H

#include <qt/Base3DView.h>

namespace aphid {

class RotationHandle;
class TranslationHandle;
class ScalingHandle;

}

class LegendreDFTest;

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
	
signals:

private:
	
private:
	LegendreDFTest* m_legen;
	aphid::Matrix44F m_space;
	aphid::Ray m_incident;
    
};

#endif
