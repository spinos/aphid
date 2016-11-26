#ifndef INST_GLWIDGET_H
#define INST_GLWIDGET_H
#include <QGLWidget>
#include <Base3DView.h>

namespace aphid {
class TriangleGeodesicSphere;
class GlslInstancer;
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
    aphid::GlslInstancer * m_instancer;
    
};

#endif
