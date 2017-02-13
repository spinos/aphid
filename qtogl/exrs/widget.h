#ifndef EXRS_WIDGET_H
#define EXRS_WIDGET_H

#include <Base3DView.h>
#include <math/ATypes.h>

namespace aphid {

class ExrImage;

}

class GLWidget : public aphid::Base3DView
{
    Q_OBJECT

public:
    GLWidget(const std::string & fileName, QWidget *parent = 0);
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
	void sampleImage();
	
private:
    aphid::ExrImage * m_sampler;
#define NUM_SMP 60000
	aphid::Float3 m_pos[NUM_SMP];
	aphid::Float4 m_col[NUM_SMP];
	
};

#endif
