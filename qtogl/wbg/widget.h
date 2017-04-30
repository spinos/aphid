/*
 *  widget.h
 *  world block grid
 *
 */

#ifndef WBG_GLWIDGET_H
#define WBG_GLWIDGET_H

#include <qt/Base3DView.h>
#include "ModifyHeightField.h"
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

class LandBlock;

}

template<typename T1, typename T2>
class DrawGraph;

struct DistanceNode;
struct IDistanceEdge;

}

struct TFTNode {
    float _distance;
};

class GLWidget : public aphid::Base3DView, public aphid::DrawTetrahedron, public ModifyHeightField
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
	virtual void keyPressEvent(QKeyEvent *event);
	virtual void keyReleaseEvent(QKeyEvent *event);
    virtual void resetPerspViewTransform();
	virtual void resetOrthoViewTransform();
	
public slots:
	void recvToolAction(int x);
	void recvHeightFieldAdd();
	void recvHeightFieldSel(int x);
	void recvHeightFieldTransformTool(int x);
	
private:
    void drawTetraMesh();
    void drawTriangulation();
	void toggleDrawTriangulationWire();
	
private slots:

private:

typedef aphid::DrawGraph<aphid::DistanceNode, aphid::IDistanceEdge > FieldDrawerT;
    FieldDrawerT * m_fieldDrawer;
    aphid::ttg::LandBlock * m_landBlk;
	bool m_doDrawTriWire;
	
	int m_showComponent;
	
};

#endif
