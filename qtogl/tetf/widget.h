#ifndef VTG_GLWIDGET_H
#define VTG_GLWIDGET_H

#include <QGLWidget>
#include <Base3DView.h>

#include <ttg/TetrahedronGrid.h>
#include <ogl/DrawTetrahedron.h>

class GLWidget : public aphid::Base3DView, public aphid::DrawTetrahedron
{
    Q_OBJECT

public:

    GLWidget(QWidget *parent = 0);
    ~GLWidget();
	
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
    void drawWiredGrid();
    void drawSolidGrid();
    
private slots:

private:
#define G_ORDER 5
	typedef aphid::TetrahedronGrid<float, G_ORDER> GridT;
	GridT * m_tg;
};

#endif
