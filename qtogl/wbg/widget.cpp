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
#include "../cactus.h"

using namespace aphid;

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	m_doDrawTriWire = false;
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
	usePerspCamera();
    
	m_landBlk = new ttg::LandBlock;
	
    m_fieldDrawer = new FieldDrawerT;
	
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{
    m_fieldDrawer->initGlsl();
}

void GLWidget::clientDraw()
{
	//updatePerspectiveView();
	//getDrawer()->frustum(perspectiveView()->frustum() );
	
	getDrawer()->m_wireProfile.apply();
	getDrawer()->setColor(.125f, .125f, .5f);
    
    //draw3LevelGrid(4);
	//drawSampleGrid();
	//drawSamples();
    
    getDrawer()->m_markerProfile.apply();
    getDrawer()->setColor(0.f, .35f, .13f);
    
	drawTetraMesh();
    //drawField();
	//drawTriangulation();
	if(m_doDrawTriWire) {
	//	drawTriangulationWire();
	}
	//drawCoarseGrid();
    
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

void GLWidget::toggleDrawTriangulationWire()
{ m_doDrawTriWire = !m_doDrawTriWire; }

void GLWidget::clientSelect(Vector3F & origin, Vector3F & ray, Vector3F & hit)
{
}
//! [9]

void GLWidget::clientDeselect()
{
}

//! [10]
void GLWidget::clientMouseInput(Vector3F & stir)
{
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
	