#ifndef BRH_WIDGET_H
#define BRH_WIDGET_H

#include <qt/Base3DView.h>
#include <ogl/DrawGlyph.h>
#include <IntersectionContext.h>
#include <boost/scoped_array.hpp>

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
	bool intersect(const aphid::Ray * incident);
	int closestNodeOnFace(int i) const;
	void drawAnchorNodes();
	void calcDistanceToRoot();
	
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
	int m_rootNodeInd;
	
	boost::scoped_array<float> m_dist2Root;
	boost::scoped_array<float> m_dysCols;
	
	aphid::topo::GeodesicDistance* m_gedis;
	
	static const float DspRootColor[3];
	static const float DspTipColor[8][3];
	
};

#endif
