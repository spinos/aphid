#ifndef VTG_GLWIDGET_H
#define VTG_GLWIDGET_H

#include <QGLWidget>
#include <Base3DView.h>

#include "TetrahedronGrid.h"


namespace ttg {

class vtgWidget : public aphid::Base3DView
{
    Q_OBJECT

public:

    vtgWidget(QWidget *parent = 0);
    ~vtgWidget();
	
protected:    
    virtual void clientInit();
    virtual void clientDraw();
    virtual void clientSelect(aphid::Vector3F & origin, aphid::Vector3F & ray, aphid::Vector3F & hit);
    virtual void clientDeselect();
    virtual void clientMouseInput(aphid::Vector3F & stir);
	virtual void keyPressEvent(QKeyEvent *event);
	virtual void keyReleaseEvent(QKeyEvent *event);
    
public slots:
		
private:

private slots:

private:
#define GORDER 5
	typedef aphid::TetrahedronGrid<float, GORDER> GridT;
	GridT * m_tg;
};

}
#endif
