#ifndef BRT_WIDGET_H
#define BRT_WIDGET_H

#include <Base3DView.h>
#include <math/ATypes.h>

namespace aphid {

class RotationHandle;

}

class FeatherMesh;
class FeatherDeformer;

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
	void drawFeather();

	aphid::Matrix44F m_space;
	aphid::Ray m_incident;
	aphid::RotationHandle * m_roth;
	
	FeatherMesh * m_mesh;
	FeatherDeformer * m_deform;
	
};

#endif
