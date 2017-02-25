#ifndef VTG_GLWIDGET_H
#define VTG_GLWIDGET_H

#include <QGLWidget>
#include <Base3DView.h>

#include <ogl/DrawTetrahedron.h>

namespace aphid {

namespace cvx {
class Triangle;
}

namespace sdb {

template<typename T>
class VectorArray;

template<typename T>
class WorldGrid2;

template<typename T, typename Tv>
class Array;

class LodSampleCache;

class LodGrid;
class LodCell;
class LodNode;

template<typename T1, typename T2, typename T3>
class GridClosestToPoint;

}

template<int I>
class KdNNode;

template<typename T1, typename T2>
class KdNTree;

template <typename Tv, typename Tg>
class TetraGridTriangulation;

class ATriangleMesh;

namespace ttg {

class AdaptiveBccGrid3;

template<typename T>
class TetrahedronDistanceField;

template<typename T>
class GenericTetraGrid;

template <typename Tv, typename Tg>
class MassiveTetraGridTriangulation;

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
    void drawTetraMesh();
    void draw3LevelGrid(int level);
	void drawSampleGrid();
	void drawSamples();
    void drawField();
    void drawTriangulation();
	void drawCoarseGrid();
    
private slots:

private:
typedef aphid::KdNTree<aphid::cvx::Triangle, aphid::KdNNode<4> > TreeTyp;
	TreeTyp * m_tree;
	aphid::sdb::VectorArray<aphid::cvx::Triangle > * m_triangles;

typedef aphid::ttg::AdaptiveBccGrid3 GridTyp;    
    GridTyp * m_grid;
    
typedef aphid::ttg::GenericTetraGrid<TFTNode > TetGridTyp;
    TetGridTyp * m_tetg;
    
typedef aphid::ttg::TetrahedronDistanceField<TetGridTyp > FieldTyp;
    
typedef aphid::DrawGraph<aphid::DistanceNode, aphid::IDistanceEdge > FieldDrawerT;
    FieldDrawerT * m_fieldDrawer;

typedef aphid::ttg::MassiveTetraGridTriangulation<TFTNode, TetGridTyp > MesherT;
    MesherT * m_mesher;
    
    aphid::ATriangleMesh * m_frontMesh;
	
typedef aphid::sdb::WorldGrid2<aphid::sdb::LodSampleCache > SampGridTyp;
	SampGridTyp * m_sampg;
	
typedef aphid::sdb::LodGrid CoarseGridType;
	
typedef aphid::sdb::GridClosestToPoint<CoarseGridType, aphid::sdb::LodCell, aphid::sdb::LodNode > SelGridTyp;
	
};

#endif
