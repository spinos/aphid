#ifndef INST_GLWIDGET_H
#define INST_GLWIDGET_H

#include <qt/Base3DView.h>
#include <ogl/DrawParticle.h>

namespace aphid {
class EbpSphere;

namespace smp {
class SampleFilter;
}

class GlslInstancer;

}

class GLWidget : public aphid::Base3DView, public aphid::DrawParticle
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
	void initInst();
    void drawSamples();
    void drawTest();
    
private:
	aphid::EbpSphere * m_grid;
	aphid::smp::SampleFilter* m_flt;
	aphid::GlslInstancer* m_inst;
	
};

#endif
