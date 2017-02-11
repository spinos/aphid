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
#include <h5/LoadElemAsset.h>
#include <sdb/LodGrid.h>
#include <ogl/DrawGridSample.h>
#include <sdb/LodGridMesher.h>

using namespace aphid;

GLWidget::GLWidget(const std::string & fileName, QWidget *parent) : Base3DView(parent)
{	
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
	usePerspCamera();
	
    m_triangles = new sdb::VectorArray<cvx::Triangle>();
	BoundingBox worldBox;
	LoadElemAsset<cvx::Triangle > loader;
	loader.loadSource(m_triangles, worldBox, fileName);
	
	KdEngine eng;
	TreeProperty::BuildProfile bf;
	bf._maxLeafPrims = 64;
	
	m_tree = new TreeTyp;
	
	float d0 = worldBox.distance(0);
	float d1 = worldBox.distance(1);
	float d2 = worldBox.distance(2);
	BoundingBox rootBox;
	rootBox.setMin(0.f, 0.f, 0.f);
	rootBox.setMax(d0, d1, d2);
	std::cout<<"\n root bbox "<<rootBox;
	
	eng.buildTree<cvx::Triangle, KdNode4, 4>(m_tree, m_triangles, rootBox, &bf);
	m_tree->setRelativeTransform(rootBox);
	
typedef IntersectEngine<cvx::Triangle, KdNode4 > FIntersectTyp;

	FIntersectTyp ineng(m_tree);
    
    const float sz0 = m_tree->getBBox().getLongestDistance() * .59f;
    
	rootBox.expand(1.f);
	
	typedef ClosestToPointEngine<cvx::Triangle, KdNode4 > FClosestTyp;
    FClosestTyp clseng(m_tree);
	
#if 0
	m_grid = new GridTyp;
    m_grid->fillBox(rootBox, sz0 );
    m_grid->subdivideToLevel<FIntersectTyp>(ineng, 0, 5);
    m_grid->build();
    
    m_tetg = new TetGridTyp;
    
    ttg::TetraMeshBuilder teter;
    teter.buildMesh(m_tetg, m_grid);
    
    m_mesher = new MesherT;
    m_mesher->setGrid(m_tetg);
    
    Vector3F rgp(0.f, 0.f, 0.f), rgn(1.f, 1.f, 1.f);
    float offset = m_grid->levelCellSize(7);
    
    FieldTyp * fld = m_mesher->field();
    
    fld->calculateDistance<FClosestTyp>(m_tetg, &clseng, rgp, rgn, offset);
    
    m_mesher->triangulate();
    
    m_frontMesh = new ATriangleMesh;
    m_mesher->dumpFrontTriangleMesh(m_frontMesh);
    m_frontMesh->calculateVertexNormals();
#endif    
    m_fieldDrawer = new FieldDrawerT;
    
	m_lodg = new LodGridTyp;
	m_lodg->fillBox(rootBox, sz0);
	m_lodg->subdivideToLevel<FIntersectTyp>(ineng, 0, 5);
	m_lodg->insertNodeAtLevel<FClosestTyp, 3 >(5, clseng);

	m_sampleDrawer = new GridSampleDrawerT(m_lodg);
	
	sdb::LodGridMesher<LodGridTyp, sdb::LodNode > lodmesher(m_lodg);
	m_l5mesh = new ATriangleMesh;
	lodmesher.buildMesh(m_l5mesh, 5);
	
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{
    m_fieldDrawer->initGlsl();
	m_sampleDrawer->initGlsl();
}

void GLWidget::clientDraw()
{
	//updatePerspectiveView();
	//getDrawer()->frustum(perspectiveView()->frustum() );
	
	getDrawer()->m_wireProfile.apply();
	getDrawer()->setColor(.125f, .125f, .5f);
#if 0
    glBegin(GL_TRIANGLES);
	for(int i=0;i<m_triangles->size();++i) {
		const cvx::Triangle * t = m_triangles->get(i);
		glVertex3fv((const GLfloat *)&t->P(0));
		glVertex3fv((const GLfloat *)&t->P(1));
		glVertex3fv((const GLfloat *)&t->P(2));
	}
	glEnd();
#endif
    //draw3LevelGrid(4);
    
    getDrawer()->m_markerProfile.apply();
    getDrawer()->setColor(.05f, .5f, .15f);
    //drawTetraMesh();
    //drawField();
	//drawTriangulation();
    //draw3LevelGrid(5);
//	drawLevelGridSamples(5);
	getDrawer()->m_surfaceProfile.apply();
    drawMesh(m_l5mesh);
}

void GLWidget::drawMesh(const ATriangleMesh * mesh)
{
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	
    glNormalPointer(GL_FLOAT, 0, (GLfloat*)mesh->vertexNormals() );
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)mesh->points() );
    glDrawElements(GL_TRIANGLES, mesh->numIndices(), GL_UNSIGNED_INT, mesh->indices() );
    
    glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

void GLWidget::drawLevelGridSamples(int level)
{
	m_sampleDrawer->drawLevelSamples(level);
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

void GLWidget::draw3LevelGrid(int level)
{    
    DrawGrid<LodGridTyp> dgd(m_lodg);
	getDrawer()->setColor(.025f, .5f, .25f);
    dgd.drawLevelCells(level-2);
    getDrawer()->setColor(.5f, .05f, .05f);
    dgd.drawLevelCells(level-1);
    getDrawer()->setColor(.5f, .25f, 0.f);
    dgd.drawLevelCells(level);
}

void GLWidget::drawField()
{
	m_fieldDrawer->drawEdge(m_mesher->field() );
    //m_fieldDrawer->drawNode(m_mesher->field() );
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
	