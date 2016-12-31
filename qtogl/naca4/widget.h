#ifndef NACA_4_DIGIT_WIDGET_H
#define NACA_4_DIGIT_WIDGET_H

#include <Base3DView.h>
#include <math/ATypes.h>

class GLWidget : public aphid::Base3DView
{
    Q_OBJECT

public:
    GLWidget(QWidget *parent = 0);
    ~GLWidget();

protected:
    virtual void clientInit();
    virtual void clientDraw();

public slots:
	void recvParam(aphid::Float3 v);
	
signals:

private:
	aphid::Float3 m_cpt;
	
};

#endif
