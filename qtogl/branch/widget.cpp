/* 
 *  branching by geodesic distance
 *  one root, multiple leaves
 *  for each vertex, calculate geodesic distance to root and each leaf
 *  select the path from root to leaf with lowest sum of distance
 */
 
#include <GeoDrawer.h>
#include <QtGui>
#include <QtOpenGL>
#include "widget.h"
#include <ogl/DrawCircle.h>
#include <ogl/RotationHandle.h>
#include <BaseCamera.h>
#include <ogl/DrawBox.h>
#include <ogl/DrawDop.h>
#include <math/AOrientedBox.h>
#include <sdb/VectorArray.h>
#include <kd/IntersectEngine.h>
#include <kd/ClosestToPointEngine.h>
#include <ogl/DrawKdTree.h>
#include <ogl/DrawGrid.h>
#include <topo/GeodesicDistance.h>
#include "../cylinder.h"

using namespace aphid;

const float GLWidget::DspRootColor[3] = {1.f, 0.f, 0.f};
const float GLWidget::DspTipColor[8][3] = {
{1.f, 1.f, 0.f},
{0.f, 1.f, 1.f},
{.8f, 0.f, .5f},
{0.f, .5f, .8f},
{.5f, .8f, 0.f},
{.4f, .3f, 1.f},
{.4f, 1.f, .3f},
{1.f, .4f, .3f},
};

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{ 
	m_interactMode = imSelectRoot;
	m_rootNodeInd = -1;
	usePerspCamera(); 
	m_triangles = new sdb::VectorArray<cvx::Triangle>();
/// prepare kd tree
	BoundingBox gridBox;
	KdEngine eng;
	eng.buildSource<cvx::Triangle, 3 >(m_triangles, gridBox,
									sCylinderMeshVertices,
									sCylinderNumTriangleIndices,
									sCylinderMeshTriangleIndices);
									
	TreeProperty::BuildProfile bf;
	bf._maxLeafPrims = 16;
	
	m_tree = new TreeTyp;
	
	eng.buildTree<cvx::Triangle, KdNode4, 4>(m_tree, m_triangles, gridBox, &bf);
	
typedef IntersectEngine<cvx::Triangle, KdNode4 > FIntersectTyp;

	FIntersectTyp ineng(m_tree);
	const float sz0 = m_tree->getBBox().getLongestDistance() * .89f;
	
typedef ClosestToPointEngine<cvx::Triangle, KdNode4 > FClosestTyp;
	
	FClosestTyp clseng(m_tree);
	
	m_gedis = new topo::GeodesicDistance;
	m_gedis->buildTriangleGraph(sCylinderNumVertices,
							sCylinderMeshVertices,
							sCylinderNumTriangleIndices / 3,
							sCylinderMeshTriangleIndices);
	m_gedis->verbose();
	
	m_dist2Root.reset(new float[sCylinderNumVertices]);
	m_dysCols.reset(new float[sCylinderNumVertices * 3]);
	
	const float defCol[3] = {0.f, .35f, .45f};
	for(int i=0;i<sCylinderNumVertices;++i) {
		memcpy(&m_dysCols[i*3], defCol, 12 );
	}
	std::cout.flush();	
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{
}

void GLWidget::clientDraw()
{
	getDrawer()->setColor(0.f, .35f, .45f);
	
	getDrawer()->m_wireProfile.apply();

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	
	glColorPointer(3, GL_FLOAT, 0, (GLfloat*)m_dysCols.get() );
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)sCylinderMeshVertices );
	glDrawElements(GL_TRIANGLES, sCylinderNumTriangleIndices, GL_UNSIGNED_INT, sCylinderMeshTriangleIndices );
	
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	
	getDrawer()->m_surfaceProfile.apply();
	//getDrawer()->m_markerProfile.apply();
	drawAnchorNodes();
}

void GLWidget::drawAnchorNodes()
{
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	
	const Vector3F* pv = (const Vector3F*)sCylinderMeshVertices;
	
	if(m_rootNodeInd > -1) {
		getDrawer()->setSurfaceColor(DspRootColor[0], DspRootColor[1], DspRootColor[2]);
		const Vector3F pn = pv[m_rootNodeInd];
		glPushMatrix();
		glTranslatef(pn.x, pn.y, pn.z);
		drawAGlyph();
		glPopMatrix();
	}
	
	const int ntip = m_tipIndices.size();
	for(int i=0;i<ntip;++i) {
	    const float* ci = DspTipColor[i&7];
	    getDrawer()->setSurfaceColor(ci[0], ci[1], ci[2]);
		const Vector3F pn = pv[m_tipIndices[i]];
		glPushMatrix();
		glTranslatef(pn.x, pn.y, pn.z);
		drawAGlyph();
		glPopMatrix();
	}
	
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

void GLWidget::clientSelect(QMouseEvent *event)
{
	const Ray* incr = getIncidentRay();
	
	switch(m_interactMode) {
		case imSelectRoot:
			selectRootNode(incr);
		break;
		case imSelectTip:
			selectTipNode(incr);
		break;
		default:
		;
	}
	update();
}

void GLWidget::clientDeselect(QMouseEvent *event)
{
	switch(m_interactMode) {
		case imSelectRoot:
			calcDistanceToRoot();
			clearAllTips();
		break;
		case imSelectTip:
			calcDistanceToTip();
			segmentRootToTip();
		break;
		default:
		;
	}
	update();
}

void GLWidget::clientMouseInput(QMouseEvent *event)
{
	const Ray* incr = getIncidentRay();
	
	switch(m_interactMode) {
		case imSelectRoot:
			selectRootNode(incr);
		break;
		case imSelectTip:
			moveTipNode(incr);
		break;
		default:
		;
	}
	update();
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	switch (e->key()) {
		case Qt::Key_R:
			m_interactMode = imSelectRoot;
			qDebug()<<" begin select root";
			break;
		case Qt::Key_T:
			m_interactMode = imSelectTip;
			qDebug()<<" begin select tip";
			break;
		default:
		;
	}
	Base3DView::keyPressEvent(e);
}
	
void GLWidget::selectRootNode(const Ray* incident)
{
	if(!intersect(incident))
		return;
	m_rootNodeInd = closestNodeOnFace(m_intersectCtx.m_componentIdx);
	std::cout<<"\n select root node "<<m_rootNodeInd;
	std::cout.flush();
}

void GLWidget::selectTipNode(const aphid::Ray * incident)
{
    if(!intersect(incident))
		return;
	int itip = closestNodeOnFace(m_intersectCtx.m_componentIdx);
	std::cout<<"\n select tip node "<<itip
	<<"\n distance to root "<<m_dist2Root[itip];
	std::cout.flush();
	m_tipIndices.push_back(itip);
	FltArrTyp* ad = new FltArrTyp;
	ad->reset(new float[sCylinderNumVertices] );
	m_dist2Tip.push_back(ad);
}

void GLWidget::moveTipNode(const aphid::Ray * incident)
{
    if(m_tipIndices.size() < 1)
        return;
    if(!intersect(incident))
		return;
	int itip = closestNodeOnFace(m_intersectCtx.m_componentIdx);
	std::cout<<"\n reselect tip node "<<itip
	<<"\n distance to root "<<m_dist2Root[itip];
	std::cout.flush();
	m_tipIndices.back() = itip;
}

int GLWidget::closestNodeOnFace(int i) const
{
	if(i>=m_triangles->size()) 
		return -1;
		
	const cvx::Triangle * t = m_triangles->get(i);
	
	int ni = -1;
	float minD = 1e8f;
	
	for(int j=0;j<3;++j) {
		float d = m_intersectCtx.m_hitP.distanceTo(t->P(j));
		if(minD > d) {
			ni = j;
			minD = d;
		}
	}
	
	if(ni<0)
		return ni;
		
	return sCylinderMeshTriangleIndices[i * 3 + ni];
	
}

bool GLWidget::intersect(const aphid::Ray * incident)
{
	m_intersectCtx.reset(*incident);
	KdEngine engine;
	try {
	engine.intersect<cvx::Triangle, KdNode4>(m_tree, &m_intersectCtx );
	} catch(const char * ex) {
	    std::cerr<<" intersect caught: "<<ex;
	} catch(...) {
	    std::cerr<<" intersect caught something";
	}
	return m_intersectCtx.m_success;
}

void GLWidget::calcDistanceToRoot()
{
	if(m_rootNodeInd < 0)
		return;
		
	m_gedis->calaculateDistanceTo(m_dist2Root.get(), m_rootNodeInd);
	
	const float& maxD = m_gedis->maxDistance();
	std::cout<<"\n max distance "<<maxD;
	for(int i=0;i<sCylinderNumVertices;++i) {
		float* ci = &m_dysCols[i*3];
		if(m_dist2Root[i] > maxD) {
			memset(ci, 0, 12 );
		} else {
			ci[1] = m_dist2Root[i] / maxD;
			ci[0] = 1.f - ci[1];
			ci[2] = 0.f;
		}
	}
	std::cout.flush();
}

void GLWidget::calcDistanceToTip()
{
    if(m_tipIndices.size() < 1)
        return;
    
    FltArrTyp* dest = m_dist2Tip.back();
    m_gedis->calaculateDistanceTo(dest->get(), m_tipIndices.back());
	
}

void GLWidget::segmentRootToTip()
{
    if(m_tipIndices.size() < 1)
        return;
    
    const int itip = m_tipIndices.size() - 1;
    const float* tipCol = DspTipColor[itip & 7];
    const int& tipNode = m_tipIndices.back();
    const float* distT = m_dist2Tip.back()->get();
    const float* distR = m_dist2Root.get();
    const float& pthl = distR[tipNode];
    std::cout<<"\n path length "<<pthl;
    int count = 0;
    for(int i=0;i<sCylinderNumVertices;++i) {
        float diff = (distT[i] + distR[i] - pthl) / pthl;
		if(diff > -.1f && diff < .1f) {
		    count++;
		    float* ci = &m_dysCols[i*3];
		    memcpy(ci, tipCol, 12);
		}
	}
	std::cout<<"\n found n node "<<count;
	std::cout.flush();
}

void GLWidget::clearAllTips()
{
    if(m_tipIndices.size() < 1)
        return;
    m_tipIndices.clear();
    std::deque<FltArrTyp* >::iterator it = m_dist2Tip.begin();
    for(;it!=m_dist2Tip.end();++it) {
        delete *it;
    }
    m_dist2Tip.clear();
}

