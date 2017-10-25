/*
 *  branching by geodesic distance
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
#include <topo/GeodesicPath.h>
#include "../cylinder.h"

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{ 
	m_interactMode = imSelectRoot;
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
	
	m_gedpath = new topo::GeodesicPath;
	m_gedpath->create(sCylinderNumVertices);
	
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
	
	glColorPointer(3, GL_FLOAT, 0, (GLfloat*)m_gedpath->dysCols() );
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)sCylinderMeshVertices );
	glDrawElements(GL_TRIANGLES, sCylinderNumTriangleIndices, GL_UNSIGNED_INT, sCylinderMeshTriangleIndices );
	
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	
	getDrawer()->m_surfaceProfile.apply();
	//getDrawer()->m_markerProfile.apply();
	drawAnchorNodes();
	drawSkeleton();
}

void GLWidget::drawAnchorNodes()
{
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	
	const Vector3F* pv = (const Vector3F*)sCylinderMeshVertices;
	
	const int nroot = m_gedpath->numRoots();
	for(int i=0;i<nroot;++i) {
		const float* ci = m_gedpath->dspRootColR();
		getDrawer()->setSurfaceColor(ci[0], ci[1], ci[2]);
		const Vector3F& pn = pv[m_gedpath->rootNodeIndices()[i]];
		glPushMatrix();
		glTranslatef(pn.x, pn.y, pn.z);
		drawAGlyph();
		glPopMatrix();
	}
	
	const int ntip = m_gedpath->numTips();
	for(int i=0;i<ntip;++i) {
	    const float* ci = m_gedpath->dspTipColR(i);
	    getDrawer()->setSurfaceColor(ci[0], ci[1], ci[2]);
		const Vector3F& pn = pv[m_gedpath->tipNodeIndices()[i]];
		glPushMatrix();
		glTranslatef(pn.x, pn.y, pn.z);
		drawAGlyph();
		glPopMatrix();
	}
	
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

void GLWidget::drawSkeleton()
{
	const int& nj = m_gedpath->numJoints();
	if(nj < 1)
		return;
	
	getDrawer()->setSurfaceColor(.7f, .8f, .7f);
	
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
		
	for(int i=0;i<nj;++i) {
		const Vector3F& pn = m_gedpath->jointPos()[i];
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
			m_gedpath->clearAllPath();
		break;
		case imSelectTip:
			calcDistanceToTip();
			m_gedpath->findPathToTip();
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
			moveRootNode(incr);
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
		case Qt::Key_Y:
			buildPaths();
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
	int iroot = closestNodeOnFace(m_intersectCtx.m_componentIdx);
	std::cout<<"\n select root node "<<iroot;
	std::cout.flush();
	m_gedpath->addRoot(iroot);
}

void GLWidget::selectTipNode(const aphid::Ray * incident)
{
    if(!intersect(incident))
		return;
	int itip = closestNodeOnFace(m_intersectCtx.m_componentIdx);
	std::cout<<"\n select tip node "<<itip;
	std::cout.flush();
	m_gedpath->addTip(itip);
}

void GLWidget::moveRootNode(const aphid::Ray * incident)
{
	if(!m_gedpath->hasRoot())
        return;
    if(!intersect(incident))
		return;
	int iroot = closestNodeOnFace(m_intersectCtx.m_componentIdx);
	std::cout<<"\n reselect root node "<<iroot;
	std::cout.flush();
	m_gedpath->setLastRootNodeIndex(iroot);
}

void GLWidget::moveTipNode(const aphid::Ray * incident)
{
    if(!m_gedpath->hasTip() )
        return;
    if(!intersect(incident))
		return;
	int itip = closestNodeOnFace(m_intersectCtx.m_componentIdx);
	std::cout<<"\n reselect tip node "<<itip;
	std::cout.flush();
	m_gedpath->setLastTipNodeIndex(itip);
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
	if(!m_gedpath->hasRoot() )
		return;
		
	m_gedis->setNodeValues<std::deque<int> >(m_gedpath->rootNodeIndices(), 0.f);
	m_gedis->calaculateDistance(m_gedpath->distanceToRoot() );
	
	const float& maxD = m_gedis->maxDistance();
	std::cout<<"\n max distance "<<maxD;
	
	m_gedpath->colorByDistanceToRoot(maxD);
	
	std::cout.flush();
}

void GLWidget::calcDistanceToTip()
{
    if(!m_gedpath->hasTip() )
        return;
    
    m_gedis->calaculateDistanceTo(m_gedpath->distanceToLastTip(), 
				m_gedpath->lastTipNodeIndex() );
	
}

void GLWidget::buildPaths()
{
	if(!m_gedpath->hasRoot() )
		return;
	if(!m_gedpath->hasTip() )
        return;
		
	const float& unitD = m_gedis->maxDistance() * .09f;
	std::cout<<"\n unit distance "<<unitD;
	
	const Vector3F* pv = (const Vector3F*)sCylinderMeshVertices;
	m_gedpath->build(unitD, pv);
	std::cout.flush();
}
