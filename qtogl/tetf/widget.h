#ifndef VTG_GLWIDGET_H
#define VTG_GLWIDGET_H

#include <QGLWidget>
#include <Base3DView.h>

#include <ttg/TetraGridTriangulation.h>
#include <ogl/DrawTetrahedron.h>

namespace aphid {

namespace ttg {

template<typename T>
class TetrahedronDistanceField;

}

template<typename T1, typename T2>
class DrawGraph;

struct DistanceNode;
struct IDistanceEdge;

}

struct TFTNode {
    float _distance;
};

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
    void drawGridEdges();
    void drawField();
    void drawTriangulation();
               
private slots:

private:
typedef aphid::TetraGridTriangulation<TFTNode, 5> MesherT;
    MesherT m_mesher;
	MesherT::GridT * m_grd;
    aphid::ttg::TetrahedronDistanceField<MesherT::GridT > * m_field;

typedef aphid::DrawGraph<aphid::DistanceNode, aphid::IDistanceEdge > FieldDrawerT;
    FieldDrawerT * m_fieldDrawer;
    
};

#endif
