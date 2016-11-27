#ifndef INST_GLWIDGET_H
#define INST_GLWIDGET_H
#include <QGLWidget>
#include <Base3DView.h>
#include <AllMath.h>

namespace aphid {
class TriangleGeodesicSphere;
class GlslLegacyInstancer;
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
    aphid::TriangleGeodesicSphere * m_sphere;
    aphid::GlslLegacyInstancer * m_instancer;
    aphid::Float4 * m_particles;
    int m_numParticles;
    
};

#endif
