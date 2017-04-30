/*
 *  widget.cpp
 *  world block grid
 *
 */
#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include <PerspectiveView.h>
#include "widget.h"
#include <GeoDrawer.h>
#include <ogl/DrawGrid.h>
#include <AllTtg.h>
#include <ogl/DrawGraph.h>
#include <kd/IntersectEngine.h>
#include <kd/ClosestToPointEngine.h>
#include <ogl/DrawGraph.h>
#include <ogl/DrawSample.h>
#include <sdb/WorldGrid2.h>
#include <sdb/LodSampleCache.h>
#include <sdb/GridClosestToPoint.h>
#include <img/HeightField.h>
#include "wbg_common.h"

using namespace aphid;

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	m_doDrawTriWire = false;
	perspCamera()->setFarClipPlane(30000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(30000.f);
	orthoCamera()->setNearClipPlane(1.f);
	usePerspCamera();
	resetView();
	
	img::HeightField::SetGlobalProfile(.5f, -200.f, 450.f);
	
	m_landBlk = new ttg::LandBlock;
	m_landBlk->processHeightField();
	m_landBlk->triangulate();
	
    m_fieldDrawer = new FieldDrawerT;
	m_showComponent = wbg::scDefault;
	
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{
    m_fieldDrawer->initGlsl();
}

void GLWidget::clientDraw()
{
    getDrawer()->m_markerProfile.apply();
    getDrawer()->setColor(0.f, .35f, .13f);
	//drawTetraMesh();

	getDrawer()->m_surfaceProfile.apply();
	drawTriangulation();
	if(m_doDrawTriWire) {
		getDrawer()->m_wireProfile.apply();
		getDrawer()->setColor(1,.7,0);
		drawTriangulation();
	}
	
	if(m_showComponent >= wbg::scHeightField) {
		getDrawer()->m_markerProfile.apply();
		drawHeightField();
	}
	
}

void GLWidget::drawTetraMesh()
{
	const ttg::LandBlock::TetGridTyp * g = m_landBlk->grid();
	
    const int nt = g->numCells();
    cvx::Tetrahedron atet;
    
    glBegin(GL_LINES);
    for(int i=0;i<nt;++i) {
        g->getCell(atet, i);
        
        glVertex3fv((const GLfloat *)&atet.X(0) );
        glVertex3fv((const GLfloat *)&atet.X(1) );
        
        glVertex3fv((const GLfloat *)&atet.X(1) );
        glVertex3fv((const GLfloat *)&atet.X(2) );
        
        glVertex3fv((const GLfloat *)&atet.X(2) );
        glVertex3fv((const GLfloat *)&atet.X(0) );
        
        glVertex3fv((const GLfloat *)&atet.X(0) );
        glVertex3fv((const GLfloat *)&atet.X(3) );
        
        glVertex3fv((const GLfloat *)&atet.X(1) );
        glVertex3fv((const GLfloat *)&atet.X(3) );
        
        glVertex3fv((const GLfloat *)&atet.X(2) );
        glVertex3fv((const GLfloat *)&atet.X(3) );
        
    }
    glEnd();
    
}

void GLWidget::drawTriangulation()
{
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	const ATriangleMesh * fm = m_landBlk->frontMesh();

	const unsigned nind = fm->numIndices();
	const unsigned * inds = fm->indices();
	const Vector3F * pos = fm->points();
	const Vector3F * nml = fm->vertexNormals();
	
	glNormalPointer(GL_FLOAT, 0, (GLfloat*)nml );
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)pos );
	glDrawElements(GL_TRIANGLES, nind, GL_UNSIGNED_INT, inds);

	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

void GLWidget::toggleDrawTriangulationWire()
{ m_doDrawTriWire = !m_doDrawTriWire; }

void GLWidget::resetPerspViewTransform()
{
static const float mm[16] = {1.f, 0.f, 0.f, 0.f,
					0.f, 0.8660254f, -0.5f, 0.f,
					0.f, 0.5f, 0.8660254f, 0.f,
					0.f, 2000.f, 3464.101616f, 1.f};
	Matrix44F mat(mm);
	perspCamera()->setViewTransform(mat, 4000.f);
}

void GLWidget::resetOrthoViewTransform()
{
static const float mm1[16] = {1.f, 0.f, 0.f, 0.f,
					0.f, 0.f, -1.f, 0.f,
					0.f, 1.f, 0.f, 0.f,
					0.f, 20000.f, 0.f, 1.f};
	Matrix44F mat(mm1);
	orthoCamera()->setViewTransform(mat, 20000.f);
	orthoCamera()->setHorizontalAperture(8000.f);
}

void GLWidget::clientSelect(QMouseEvent *event)
{
	const Ray * incident = getIncidentRay();
	switch (heightFieldToolFlag() ) {
		case wbg::hfcMove :
		case wbg::hfcRotate :
		case wbg::hfcResize :
			beginModifyHeightField(incident);
		break;
		default:
		;
	}
}

void GLWidget::clientDeselect(QMouseEvent *event)
{
	switch (heightFieldToolFlag() ) {
		case wbg::hfcMove :
		case wbg::hfcRotate :
		case wbg::hfcResize :
			endModifyeHeightField();
		break;
		default:
		;
	}
}

void GLWidget::clientMouseInput(QMouseEvent *event)
{
	const Ray * incident = getIncidentRay();
	switch (heightFieldToolFlag() ) {
		case wbg::hfcMove :
			doMoveHeightField(incident);
		break;
		case wbg::hfcRotate :
			doRotateHeightField(incident);
		break;
		case wbg::hfcResize :
			doResizeHeightField(incident);
		break;
		default:
		;
	}
	update();
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	switch (e->key()) {
		case Qt::Key_W:
			toggleDrawTriangulationWire();
			break;
		case Qt::Key_N:
			break;
		default:
			break;
	}
	Base3DView::keyPressEvent(e);
}

void GLWidget::keyReleaseEvent(QKeyEvent *event)
{
	Base3DView::keyReleaseEvent(event);
}
	
void GLWidget::recvToolAction(int x)
{
	if(x == wbg::actViewTop) {
		resetOrthoViewTransform();
		useOrthoCamera(false);
		m_showComponent = wbg::scHeightField;
		update();
	} else if (x == wbg::actViewPersp) {
		resetPerspViewTransform();
		usePerspCamera(false);
		m_showComponent = wbg::scDefault;
		m_landBlk->rebuild();
		update();
	} else {
		qDebug()<<" unknown tool action "<<x;
	}
}

void GLWidget::recvHeightFieldAdd()
{
	if (m_showComponent == wbg::scDefault) {
		resetOrthoViewTransform();
		useOrthoCamera(false);
		m_showComponent = wbg::scHeightField;
	}
	update();
}

void GLWidget::recvHeightFieldSel(int x)
{
	selectHeightField(x);
	update();
}

void GLWidget::recvHeightFieldTransformTool(int x)
{ setHeightFieldToolFlag(x); }
