#ifndef GP_DF_WIDGET_H
#define GP_DF_WIDGET_H

#include <Base3DView.h>

namespace aphid {

template<typename T>
class DenseMatrix;

namespace gpr {
template<typename T>
class GPInterpolate;

}

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
	void recvXValue(QPointF vx);
	
signals:

private:
	aphid::gpr::GPInterpolate<float> * m_interp;
	
};

#endif
