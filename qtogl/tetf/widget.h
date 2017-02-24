#ifndef VTG_GLWIDGET_H
#define VTG_GLWIDGET_H

#include <QGLWidget>
#include <Base3DView.h>

#include <ogl/DrawTetrahedron.h>
#include <sdb/Sequence.h>

namespace aphid {

namespace cvx {
class Triangle;
}

namespace sdb {

class LodCell;
class LodGrid;
class LodNode;

template<typename T1, typename T2, typename T3>
class GridClosestToPoint;

}

template<typename T1, typename T2, typename T3>
class PrimInd;

template <typename Tv, int N>
class TetrahedronGrid;

template <typename Tv, typename Tg>
class TetraGridTriangulation;

class ATriangleMesh;

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
    void drawGround();
    void drawWiredGrid();
    void drawSolidGrid();
    void drawGridEdges();
    void drawField();
    void drawCellCut();
    void drawTriangulation();
               
private slots:

private:

typedef aphid::TetrahedronGrid<TFTNode, 5 > GridT;
    GridT * m_grd[2];
    
typedef aphid::TetraGridTriangulation<TFTNode, GridT > MesherT;
    MesherT * m_mesher[2];
	
typedef aphid::DrawGraph<aphid::DistanceNode, aphid::IDistanceEdge > FieldDrawerT;
    FieldDrawerT * m_fieldDrawer;
    
    std::vector<aphid::cvx::Triangle * > m_ground;
    aphid::sdb::Sequence<int> m_sels;

typedef aphid::PrimInd<aphid::sdb::Sequence<int>, std::vector<aphid::cvx::Triangle * >, aphid::cvx::Triangle > TIntersect;
	
    aphid::ATriangleMesh * m_frontMesh[2];

typedef aphid::sdb::LodGrid LodGridTyp; 
	LodGridTyp * m_lodg;

typedef aphid::sdb::GridClosestToPoint<LodGridTyp, aphid::sdb::LodCell, aphid::sdb::LodNode > SelGridTyp;
	SelGridTyp * m_selGrd;
	
};

#endif
