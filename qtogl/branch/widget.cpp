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
#include <topo/GeodesicSkeleton.h>
#include <topo/JointPiece.h>
#include "../cylinder.h"

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{ 
	m_selVertex = 0;
	m_interactMode = imSelectSeed;
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
	
	m_skeleton = new topo::GeodesicSkeleton;
	m_skeleton->createFromTriangles(sCylinderNumVertices,
							sCylinderMeshVertices,
							sCylinderMeshNormals,
							sCylinderNumTriangleIndices / 3,
							sCylinderMeshTriangleIndices);
	m_skeleton->verbose();
	
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
#if 1
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	
	glColorPointer(3, GL_FLOAT, 0, (GLfloat*)m_skeleton->dysCols() );
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)sCylinderMeshVertices );
	glDrawElements(GL_TRIANGLES, sCylinderNumTriangleIndices, GL_UNSIGNED_INT, sCylinderMeshTriangleIndices );
	
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
#else
	drawCurvature();
#endif
	
	getDrawer()->m_surfaceProfile.apply();
	drawAnchorNodes();
	drawSkeleton();
	getDrawer()->m_markerProfile.apply();
}

void GLWidget::drawAnchorNodes()
{
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	
	const Vector3F* pv = (const Vector3F*)sCylinderMeshVertices;
	
	const int ntip = m_skeleton->numRegions();
	
	for(int i=0;i<ntip;++i) {
	    const float* ci = m_skeleton->dspRegionColR(i);
	    getDrawer()->setSurfaceColor(ci[0], ci[1], ci[2]);
		const Vector3F& pn = pv[m_skeleton->siteNodeIndex(i)];
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
	const int& nr = m_skeleton->numPieces();
	if(nr < 1)
		return;
	
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
		
	for(int i=0;i<nr;++i) {
		const float* coli = m_skeleton->dspRegionColR(i);
		getDrawer()->setSurfaceColor(coli[0], coli[1], coli[2]);
		
		const topo::JointPiece& piecei = m_skeleton->getPiece(i);
		const int& nj = piecei.numJoints();
		
		for(int j=0;j<nj;++j) {
			const topo::JointData& jd = piecei.joints()[j];
			
			glPushMatrix();
			glTranslatef(jd._posv[0], jd._posv[1], jd._posv[2]);
			glScalef(.3f, .3f, .3f);
			drawAGlyph();
			glPopMatrix();
		}
		
	}
	
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	
	for(int i=0;i<nr;++i) {
		const float* coli = m_skeleton->dspRegionColR(i);
		getDrawer()->setSurfaceColor(coli[0], coli[1], coli[2]);
		
		const topo::JointPiece& piecei = m_skeleton->getPiece(i);
		const int& nj = piecei.numJoints();
		
		for(int j=0;j<nj;++j) {
			const topo::JointData& jd = piecei.joints()[j];
			const topo::JointData* jp = jd._parent;
			if(jp) {
				getDrawer()->arrow(Vector3F(jp->_posv),
							Vector3F(jd._posv));
			}
		}
		
	}
}

void GLWidget::draw1Ring()
{
	int nv = 0;
	const int* vj;
	m_skeleton->getVij(nv, vj, m_selVertex);
	glColor3f(1.f, 1.f, 1.f);
	glBegin(GL_LINES);
	for(int i=0;i<nv-1;++i) {
		glVertex3fv((const GLfloat*)&m_skeleton->nodes()[vj[i]].pos);
		glVertex3fv((const GLfloat*)&m_skeleton->nodes()[vj[i+1]].pos);
	}
	glEnd();
}

void GLWidget::drawCurvature()
{
	float pos[6];
	float col[3];
	const int& ne = m_skeleton->numEdges();
	
	glBegin(GL_LINES);
	for(int i=0;i<ne;++i) {
		m_skeleton->colorEdgeByCurvature(pos, col, i);
		glColor3fv(col);
		
		glVertex3fv(pos);
		glVertex3fv(&pos[3]);
	}
	glEnd();
	
}

void GLWidget::clientSelect(QMouseEvent *event)
{
	const Ray* incr = getIncidentRay();
	
	switch(m_interactMode) {
		case imSelectSeed:
			selectSeedNode(incr);
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
		case imSelectSeed:
			moveSeedNode(incr);
		break;
		default:
		;
	}
	update();
}

void GLWidget::clientDeselect(QMouseEvent *event)
{
	switch(m_interactMode) {
		case imSelectSeed:
			performSegmentation();
		break;
		default:
		;
	}
	update();
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	switch (e->key()) {
		case Qt::Key_T:
			m_interactMode = imSelectSeed;
			qDebug()<<" begin select seed point";
			break;
		case Qt::Key_Y:
			break;
		default:
		;
	}
	Base3DView::keyPressEvent(e);
}

void GLWidget::selectSeedNode(const aphid::Ray * incident)
{
	std::cout<<"\n select seed point";
	std::cout.flush();
    if(!intersect(incident))
		return;
	m_selVertex = closestNodeOnFace(m_intersectCtx.m_componentIdx);
	std::cout<<"\n select node "<<m_selVertex<<" as seed point";
	std::cout.flush();
	m_skeleton->addSeed(m_selVertex);
}

void GLWidget::moveSeedNode(const aphid::Ray * incident)
{
    if(m_skeleton->numRegions() < 2 )
        return;
    if(!intersect(incident))
		return;
	m_selVertex = closestNodeOnFace(m_intersectCtx.m_componentIdx);
	std::cout<<"\n reselect node "<<m_selVertex<<" as seed point";
	std::cout.flush();
	m_skeleton->setLastTipNodeIndex(m_selVertex);
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

void GLWidget::performSegmentation()
{
	bool stat = m_skeleton->findRootNode();
	if(!stat)
		return;
		
	m_skeleton->growRegions();
	m_skeleton->buildSkeleton();
}
