#ifndef HERMITE_WIDGET_H
#define HERMITE_WIDGET_H

#include <qt/Base3DView.h>

namespace aphid {

class Vector3F;

template<typename T1, typename T2>
class HermiteInterpolatePiecewise;

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

public slots:
	
signals:

private:
	aphid::HermiteInterpolatePiecewise<float, aphid::Vector3F > * m_interp;
	
};

#endif
