#ifndef ROTT_WIDGET_H
#define ROTT_WIDGET_H

#include <qt/Base3DView.h>
#include <math/ATypes.h>

namespace aphid {

class RotationHandle;
class TranslationHandle;
class ScalingHandle;

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
	
public slots:

signals:

private:
	void testBoxes();
	void testDops();
    void drawRotate();
    void drawTranslate();
	void drawScaling();
	
private:

	aphid::Matrix44F m_space;
	aphid::Ray m_incident;
	aphid::RotationHandle * m_roth;
	aphid::TranslationHandle * m_tranh;
	aphid::ScalingHandle * m_scalh;
    
};

#endif
