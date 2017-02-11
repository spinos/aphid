#ifndef TTI_GLWIDGET_H
#define TTI_GLWIDGET_H

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

class LodGrid;

class LodCell;

class LodNode;

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

}

template<typename T1, typename T2>
class DrawGraph;

struct DistanceNode;
struct IDistanceEdge;

template<typename T, typename T1, typename T2>
class DrawGridSample;

class ATriangleMesh;

}

struct TFTNode {
    float _distance;
};

class GLWidget : public aphid::Base3DView, public aphid::DrawTetrahedron
{
    Q_OBJECT

public:

    GLWidget(const std::string & fileName, QWidget *parent = 0);
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
	void drawLevelGridSamples(int level);
    void drawField();
    void drawTriangulation();
	void drawMesh(const aphid::ATriangleMesh * mesh);
    
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

typedef aphid::TetraGridTriangulation<TFTNode, TetGridTyp > MesherT;
    MesherT * m_mesher;
    
    aphid::ATriangleMesh * m_frontMesh;
     
typedef aphid::sdb::LodGrid LodGridTyp; 
	LodGridTyp * m_lodg;
	
typedef aphid::DrawGridSample<LodGridTyp, aphid::sdb::LodCell, aphid::sdb::LodNode > GridSampleDrawerT; 
	GridSampleDrawerT * m_sampleDrawer;
	
	aphid::ATriangleMesh * m_l5mesh;
	
};

#endif
