/*
 *  widget.cpp
 *  Adaptive Bcc Grid Test
 *
 */
#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include <PerspectiveView.h>
#include "widget.h"
#include <GeoDrawer.h>
#include <ogl/DrawGrid.h>
#include <ttg/AdaptiveBccGrid3.h>
#include <ttg/TetrahedronDistanceField.h>
#include <ttg/TetraMeshBuilder.h>
#include <ogl/DrawGraph.h>
#include <geom/PrimInd.h>
#include <kd/IntersectEngine.h>
#include <kd/ClosestToPointEngine.h>
#include "../cactus.h"

using namespace aphid;

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
	usePerspCamera();
	
    m_triangles = new sdb::VectorArray<cvx::Triangle>();
/// prepare kd tree
	BoundingBox gridBox;
	KdEngine eng;
	eng.buildSource<cvx::Triangle, 3 >(m_triangles, gridBox,
									sCactusMeshVertices[3],
									sCactusNumTriangleIndices,
									sCactusMeshTriangleIndices);
									
	// std::cout<<"\n kd tree source bbox"<<gridBox
	//		<<"\n n tri "<<m_triangles->size();
	
	TreeProperty::BuildProfile bf;
	bf._maxLeafPrims = 16;
	
	m_tree = new TreeTyp;
	
	eng.buildTree<cvx::Triangle, KdNode4, 4>(m_tree, m_triangles, gridBox, &bf);
	
typedef IntersectEngine<cvx::Triangle, KdNode4 > FIntersectTyp;

	FIntersectTyp ineng(m_tree);
    
    const float sz0 = m_tree->getBBox().getLongestDistance() * .99f;
    
    m_grid = new GridTyp;
    m_grid->fillBox(m_tree->getBBox(), sz0 );
    m_grid->subdivideToLevel<FIntersectTyp>(ineng, 0, 4);
    m_grid->build();
    
    m_teter = new ttg::TetraMeshBuilder;
    m_teter->buildMesh(m_grid);
    
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{
}

void GLWidget::clientDraw()
{
	//updatePerspectiveView();
	//getDrawer()->frustum(perspectiveView()->frustum() );
	
	getDrawer()->m_wireProfile.apply();
	getDrawer()->setColor(.125f, .125f, .5f);
	
    glBegin(GL_TRIANGLES);
	for(int i=0;i<m_triangles->size();++i) {
		const cvx::Triangle * t = m_triangles->get(i);
		glVertex3fv((const GLfloat *)&t->P(0));
		glVertex3fv((const GLfloat *)&t->P(1));
		glVertex3fv((const GLfloat *)&t->P(2));
	}
	glEnd();
    
    //draw3LevelGrid(4);
    getDrawer()->setColor(.05f, .5f, .15f);
    drawTetraMesh();
}

void GLWidget::drawTetraMesh()
{
    const Vector3F * pnts = m_teter->vertices(); 
    const int nt = m_teter->numTetrahedrons();
    cvx::Tetrahedron atet;
    
	KdEngine seleng;
    SphereSelectionContext selctx;
    Vector3F pcen;
    float frad;
    
    glBegin(GL_LINES);
    for(int i=0;i<nt;++i) {
        m_teter->getTetra(atet, i);
        atet.circumSphere(pcen, frad);
        
        selctx.deselect();
        selctx.reset(pcen, frad, SelectionContext::Append, true);
                
        seleng.select(m_tree, &selctx );
        if(selctx.numSelected() < 1) {
            continue;
        }
        
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

void GLWidget::draw3LevelGrid(int level)
{    
    DrawGrid<GridTyp> dgd(m_grid);
	getDrawer()->setColor(.5f, .05f, .05f);
    dgd.drawLevelCells(level-2);
    getDrawer()->setColor(.125f, .5f, .25f);
    dgd.drawLevelCells(level-1);
    getDrawer()->setColor(0.f, .1f, .5f);
    dgd.drawLevelCells(level);
}

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
		case Qt::Key_M:
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
	