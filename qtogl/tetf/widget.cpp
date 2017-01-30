#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include <PerspectiveView.h>
#include "widget.h"
#include <GeoDrawer.h>
#include <ogl/DrawGrid.h>
#include <ttg/RedBlueRefine.h>
#include <ttg/TetrahedronDistanceField.h>
#include <ogl/DrawGraph.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
	usePerspCamera();
	
}

GLWidget::~GLWidget()
{}

static const float stetvs[4][3] = {
{ -12.f, -4.f, -9.f}, 
{ 0.f, 12.f, 0.f},
{ 16.f, -2.5f, -9.5f}, 
{ 4.f, -8.f, 12.f}
};

void GLWidget::clientInit()
{
	cvx::Tetrahedron tetra;
	tetra.set(Vector3F(stetvs[0]), 
				Vector3F(stetvs[1]),
				Vector3F(stetvs[2]), 
				Vector3F(stetvs[3]) );
		
    TetrahedronGridUtil<5 > tu4;
	m_grd = new MesherT::GridT(tetra, 0);
    
    TFTNode anode;
    const int nn = m_grd->numPoints();
    std::cout << "\n n node " << nn;
    for(int i=0;i<nn;++i) {
        anode._distance = m_grd->pos(i).y  + 1.4f * sin(m_grd->pos(i).x * .2f + 1.f)
                                        - cos(m_grd->pos(i).z  * .3f - 2.f) ;
        m_grd->setValue(anode, i);
    }
    
    m_mesher.triangulate(m_grd);
    
    m_field = new ttg::TetrahedronDistanceField<MesherT::GridT >();
    m_field->buildGraph(m_grd, &m_mesher.gridEdges() );
    
    m_fieldDrawer = new FieldDrawerT;
    m_fieldDrawer->initGlsl();
}

void GLWidget::clientDraw()
{
	//updatePerspectiveView();
	//getDrawer()->frustum(perspectiveView()->frustum() );
	
	getDrawer()->m_markerProfile.apply();
	getDrawer()->setColor(.125f, .125f, .5f);
	
	//drawWiredGrid();
    //drawSolidGrid();
    drawField();
    drawTriangulation();
/*
    glEnable(GL_CULL_FACE);
    glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
    
    cvx::Tetrahedron tetra;
	tetra.set(Vector3F(stetvs[0]), 
				Vector3F(stetvs[1]),
				Vector3F(stetvs[2]), 
				Vector3F(stetvs[3]) );
    drawASolidTetrahedron(tetra);            
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
*/
}

void GLWidget::drawGridEdges()
{
    TetraGridEdgeMap<MesherT::GridT > & edges = m_mesher.gridEdges();
    
    glBegin(GL_LINES);
    edges.begin();
    while(!edges.end() ) {
        const sdb::Coord2 & k = edges.key();
        
        glVertex3fv((const float *)&m_grd->pos(k.x) );
        glVertex3fv((const float *)&m_grd->pos(k.y) );
        
        edges.next();
    }
    glEnd();

}

void GLWidget::drawField()
{
    m_fieldDrawer->drawEdge(m_field);
    m_fieldDrawer->drawNode(m_field);
}

void GLWidget::drawSolidGrid()
{
    getDrawer()->m_surfaceProfile.apply();
	
        glEnable(GL_CULL_FACE);
    glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	cvx::Tetrahedron atet;
    const int n = m_grd->numCells();
    for(int i=0;i<n;++i) {
        m_grd->getCell(atet, i);
       // glColor3f(RandomF01(), .5f, RandomF01() );
        drawAShrinkSolidTetrahedron(atet, .67f);
    }
    
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
}

void GLWidget::drawWiredGrid()
{
    glEnableClientState(GL_VERTEX_ARRAY);
	cvx::Tetrahedron atet;
    const int n = m_grd->numCells();
    for(int i=0;i<n;++i) {
        m_grd->getCell(atet, i);
        drawAWireTetrahedron(atet);
    }
    
    glDisableClientState(GL_VERTEX_ARRAY);

}

void GLWidget::drawTriangulation()
{
    int ntri = m_mesher.numFrontTriangles();

    Vector3F * trips = new Vector3F[ntri * 3];
    m_mesher.extractFrontTriangles(trips);
    
     getDrawer()->m_wireProfile.apply();
     getDrawer()->setColor(1,.7,0);
    glEnableClientState(GL_VERTEX_ARRAY);
	
    glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)trips );
    glDrawArrays(GL_TRIANGLES, 0, ntri * 3);
    
    glDisableClientState(GL_VERTEX_ARRAY);
    
    delete[] trips;
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
			//m_scene->progressForward();
			break;
		case Qt::Key_N:
			//m_scene->progressBackward();
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
	