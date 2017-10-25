#ifndef BRH_WIDGET_H
#define BRH_WIDGET_H

#include <qt/Base3DView.h>
#include <ogl/DrawGlyph.h>
#include <IntersectionContext.h>
#include <boost/scoped_array.hpp>
#include <deque>

namespace aphid {

class RotationHandle;

namespace cvx {

class Triangle;

}

namespace sdb {

template<typename T>
class VectorArray;

}

template<int I>
class KdNNode;

template<typename T1, typename T2>
class KdNTree;

namespace topo {

class GeodesicDistance;
class GeodesicPath;

}

}

class GLWidget : public aphid::Base3DView, public aphid::DrawGlyph
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
	
public slots:

signals:

private:
	void selectRootNode(const aphid::Ray * incident);
	void selectTipNode(const aphid::Ray * incident);
	void moveRootNode(const aphid::Ray * incident);
	void moveTipNode(const aphid::Ray * incident);
	bool intersect(const aphid::Ray * incident);
	int closestNodeOnFace(int i) const;
	void drawAnchorNodes();
	void drawSkeleton();
	void calcDistanceToRoot();
	void calcDistanceToTip();
	void buildPaths();
	
private:
typedef aphid::KdNTree<aphid::cvx::Triangle, aphid::KdNNode<4> > TreeTyp;
	TreeTyp * m_tree;
	aphid::sdb::VectorArray<aphid::cvx::Triangle > * m_triangles;
	
	enum InteractMode {
		imUnknown = 0,
		imSelectRoot,
		imSelectTip,
	};
	
	InteractMode m_interactMode;
	
	aphid::IntersectionContext m_intersectCtx;
	
	aphid::topo::GeodesicDistance* m_gedis;
	aphid::topo::GeodesicPath* m_gedpath;
	
};

#endif
