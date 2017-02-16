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
#include <ttg/GenericTetraGrid.h>
#include <ttg/TetraMeshBuilder.h>
#include <ttg/TetraGridTriangulation.h>
#include <ogl/DrawGraph.h>
#include <geom/PrimInd.h>
#include <kd/IntersectEngine.h>
#include <kd/ClosestToPointEngine.h>
#include <ogl/DrawGraph.h>
#include <sdb/WorldGrid2.h>
#include <sdb/LodGrid.h>
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
									sCactusMeshVertices[4],
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
    
    const float sz0 = m_tree->getBBox().getLongestDistance() * .87f;
    
    m_grid = new GridTyp;
    m_grid->fillBox(gridBox, sz0 );
    m_grid->subdivideToLevel<FIntersectTyp>(ineng, 0, 4);
    m_grid->build();
    
    m_tetg = new TetGridTyp;
    
    ttg::TetraMeshBuilder teter;
    teter.buildMesh(m_tetg, m_grid);
    
    m_mesher = new MesherT;
    m_mesher->setGrid(m_tetg);
    
typedef ClosestToPointEngine<cvx::Triangle, KdNode4 > FClosestTyp;
    FClosestTyp clseng(m_tree);
    
    Vector3F rgp(0.f, 0.f, 0.f), rgn(1.f, 1.f, 1.f);
	CalcDistanceProfile prof;
	prof.referencePoint = rgp;
	prof.direction = rgn;
	prof.offset = m_grid->levelCellSize(6);
    
    FieldTyp * fld = m_mesher->field();
    
    fld->calculateDistance<FClosestTyp>(m_tetg, &clseng, prof);
    
    m_mesher->triangulate();
    
    m_frontMesh = new ATriangleMesh;
    m_mesher->dumpFrontTriangleMesh(m_frontMesh);
    m_frontMesh->calculateVertexNormals();
    
    m_fieldDrawer = new FieldDrawerT;
    
	m_sampg = new SampGridTyp;
	m_sampg->setGridSize(sz0);
	
	BoundingBox cb; 
	Vector3F cbcen;
	m_grid->begin();
	while(!m_grid->end() ) {
		
		const sdb::Coord4 & k = m_grid->key();
		if(k.w > 0) {
			break;
		}
		
		m_grid->getCellBBox(cb, k);
		cbcen = m_grid->cellCenter(k);
		
		sdb::LodGrid * cell = m_sampg->insertCell((const float *)&cbcen );
		cell->resetBox(cb, sz0);
		cell->subdivideToLevel<FIntersectTyp>(ineng, 0, 4);
		cell->insertNodeAtLevel<FClosestTyp, 4 >(4, clseng);
	
		m_grid->next();
	}
	
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
	
    glBegin(GL_TRIANGLES);
	for(int i=0;i<m_triangles->size();++i) {
		const cvx::Triangle * t = m_triangles->get(i);
		glVertex3fv((const GLfloat *)&t->P(0));
		glVertex3fv((const GLfloat *)&t->P(1));
		glVertex3fv((const GLfloat *)&t->P(2));
	}
	glEnd();
    
    //draw3LevelGrid(4);
	drawSampleGrid();
    
    getDrawer()->m_markerProfile.apply();
    getDrawer()->setColor(.05f, .5f, .15f);
    //drawTetraMesh();
    //drawField();
    drawTriangulation();
    
}

void GLWidget::drawTriangulation()
{
    //getDrawer()->m_wireProfile.apply();
    getDrawer()->m_surfaceProfile.apply();
    getDrawer()->setColor(1,.7,0);

    const unsigned nind = m_frontMesh->numIndices();
    const unsigned * inds = m_frontMesh->indices();
    const Vector3F * pos = m_frontMesh->points();
    const Vector3F * nms = m_frontMesh->vertexNormals();
    
    glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	
    glNormalPointer(GL_FLOAT, 0, (GLfloat*)nms );
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)pos );
    glDrawElements(GL_TRIANGLES, nind, GL_UNSIGNED_INT, inds);
    
    glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

void GLWidget::drawTetraMesh()
{
    const int nt = m_tetg->numCells();
    cvx::Tetrahedron atet;
    
	KdEngine seleng;
    SphereSelectionContext selctx;
    Vector3F pcen;
    float frad;
    
    glBegin(GL_LINES);
    for(int i=0;i<nt;++i) {
        m_tetg->getCell(atet, i);
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

void GLWidget::drawSampleGrid()
{
	getDrawer()->setColor(.15f, .15f, .15f);
    
	m_sampg->begin();
	while(!m_sampg->end() ) {
	
		DrawGrid<sdb::LodGrid> dgd(m_sampg->value() );
		dgd.drawLevelCells(4);
		
		m_sampg->next();
	}
	
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

void GLWidget::drawField()
{
   // m_fieldDrawer->drawEdge(m_mesher->field() );
    m_fieldDrawer->drawNode(m_mesher->field() );
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
	