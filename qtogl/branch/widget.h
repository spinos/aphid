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

class GeodesicSkeleton;

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
	void selectSeedNode(const aphid::Ray * incident);
	void moveSeedNode(const aphid::Ray * incident);
	bool intersect(const aphid::Ray * incident);
	int closestNodeOnFace(int i) const;
	void drawAnchorNodes();
	void drawSkeleton();
	void draw1Ring();
	void buildPaths();
/// aft seed points selected
	void performSegmentation();
	
private:
typedef aphid::KdNTree<aphid::cvx::Triangle, aphid::KdNNode<4> > TreeTyp;
	TreeTyp * m_tree;
	aphid::sdb::VectorArray<aphid::cvx::Triangle > * m_triangles;
	
	enum InteractMode {
		imUnknown = 0,
		imSelectSeed,
	};
	
	InteractMode m_interactMode;
	
	aphid::IntersectionContext m_intersectCtx;
	
	aphid::topo::GeodesicSkeleton* m_skeleton;
	int m_selVertex;
};

#endif
