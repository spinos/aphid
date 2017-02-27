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
#include <ttg/MassiveTetraGridTriangulation.h>
#include <ogl/DrawGraph.h>
#include <geom/PrimInd.h>
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
    
    const float sz0 = m_tree->getBBox().getLongestDistance() * .97f;
	
	MesherT::Profile mshprof;
	mshprof.coarsGridBox = m_tree->getBBox();
	mshprof.coarseCellSize = sz0;
	mshprof.coarseGridSubdivLevel = 2;
	mshprof.fineGridSubdivLevel = 5;
	mshprof.offset = .1f;
/// quality of mesh
	mshprof.fineGridSubdivNormalDistribute = .9f;
	
    m_grid = new GridTyp;
    m_grid->fillBox(gridBox, sz0 );
	
	sdb::AdaptiveGridDivideProfle subdprof;
	subdprof.setLevels(0, 4);
	subdprof.setToDivideAllChild(true);
	
typedef ClosestToPointEngine<cvx::Triangle, KdNode4 > FClosestTyp;
    FClosestTyp clseng(m_tree);
	
    m_grid->subdivideToLevel<FIntersectTyp, FClosestTyp>(ineng, clseng, subdprof);
    m_grid->build();
    
    m_tetg = new TetGridTyp;
    
    ttg::TetraMeshBuilder teter;
    teter.buildMesh(m_tetg, m_grid);
    
    m_mesher = new MesherT;
	
	CoarseGridType coarseSampG;
	coarseSampG.fillBox(m_tree->getBBox(), sz0);
	
	subdprof.setLevels(0, 5);
	coarseSampG.subdivideToLevel<FIntersectTyp>(ineng, subdprof);
	
    coarseSampG.insertNodeAtLevel<FClosestTyp, 4 >(5, clseng);
	
	SelGridTyp coarseSampDistance(&coarseSampG);
	coarseSampDistance.setMaxSelectLevel(5);
    
    Vector3F rgp(0.f, 0.f, 0.f), rgn(1.f, 1.f, 1.f);
	CalcDistanceProfile prof;
	prof.referencePoint = rgp;
	prof.direction = rgn;
	prof.offset = 0.1f;//m_grid->levelCellSize(7);
    prof.snapDistance = 0.f;
    
	m_mesher->triangulate<FIntersectTyp, FClosestTyp>(ineng, clseng, mshprof);
    
    m_frontMesh = new ATriangleMesh;
    m_fieldDrawer = new FieldDrawerT;

#if 0
	m_sampg = new SampGridTyp;
	m_sampg->setGridSize(sz0);
	
	subdprof.setLevels(0, 4);
	subdprof.setToDivideAllChild(false);
	
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
		
		sdb::LodSampleCache * cell = m_sampg->insertCell((const float *)&cbcen );
		cell->resetBox(cb, sz0);
		cell->subdivideToLevel<FIntersectTyp>(ineng, subdprof);
		cell->insertNodeAtLevel<FClosestTyp, 4 >(4, clseng);
		cell->insertNodeAtLevel<FClosestTyp, 4 >(3, clseng);
		cell->inserNodedByAggregation(2, 2);
		cell->buildSampleCache(2,4);
		m_grid->next();
	}
#endif	
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
	//drawSampleGrid();
	//drawSamples();
    
    getDrawer()->m_markerProfile.apply();
    getDrawer()->setColor(0.f, .35f, .13f);
    
	//drawTetraMesh();
    drawField();
	drawTriangulation();
	if(m_doDrawTriWire) {
		drawTriangulationWire();
	}
	//drawCoarseGrid();
    
}

void GLWidget::drawCoarseGrid()
{
	getDrawer()->m_wireProfile.apply();
	getDrawer()->setColor(0.f,.5f,.7f);
	const GenericHexagonGrid<TFTNode> * g = m_mesher->coarseGrid();
	
	glBegin(GL_LINES);
	
	cvx::Hexagon ahexa;
	const int & nc = g->numCells();
	for(int i=0;i<nc;++i) {
		g->getCell(ahexa, i);
		
		for(int j=0;j<12;++j) {
			const Vector3F & p0 = ahexa.P(sdb::gdt::TwelveBlueBlueEdges[j][0] - 6);
			const Vector3F & p1 = ahexa.P(sdb::gdt::TwelveBlueBlueEdges[j][1] - 6);
			glVertex3fv((const float *)&p0);
			glVertex3fv((const float *)&p1);
		}
	}
	
	glEnd();
	
}

void GLWidget::drawTriangulation()
{
	getDrawer()->m_surfaceProfile.apply();
	
	const int nm = m_mesher->numFrontMeshes();
	
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	
	for(int i=0;i<nm;++i) {
		const ATriangleMesh * fm = m_mesher->frontMesh(i);

    const unsigned nind = fm->numIndices();
    const unsigned * inds = fm->indices();
    const Vector3F * pos = fm->points();
    const Vector3F * nms = fm->vertexNormals();
    
    glNormalPointer(GL_FLOAT, 0, (GLfloat*)nms );
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)pos );
    glDrawElements(GL_TRIANGLES, nind, GL_UNSIGNED_INT, inds);
	}
	
    glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

void GLWidget::drawTriangulationWire()
{
	getDrawer()->m_wireProfile.apply();
	getDrawer()->setColor(1,.7,0);
	
	glEnableClientState(GL_VERTEX_ARRAY);
	
	const int nm = m_mesher->numFrontMeshes();
	
	for(int i=0;i<nm;++i) {
		const ATriangleMesh * fm = m_mesher->frontMesh(i);

		const unsigned nind = fm->numIndices();
		const unsigned * inds = fm->indices();
		const Vector3F * pos = fm->points();
		
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)pos );
		glDrawElements(GL_TRIANGLES, nind, GL_UNSIGNED_INT, inds);
	}
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
		/*
        atet.circumSphere(pcen, frad);
        
        selctx.deselect();
        selctx.reset(pcen, frad, SelectionContext::Append, true);
                
        seleng.select(m_tree, &selctx );
        if(selctx.numSelected() < 1) {
            continue;
        }*/
        
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

void GLWidget::drawSamples()
{
	getDrawer()->m_surfaceProfile.apply();
	
	getDrawer()->setColor(1.f, 0.f, 0.f);
	
	DrawSample::Profile drprof;
	drprof.m_stride = sdb::SampleCache::DataStride;
	drprof.m_pointSize = 5.f;
	drprof.m_hasNormal = true;
	drprof.m_hasColor = false;
	
	DrawSample drs;
	drs.begin(drprof);
	const int dlevel = 4;
	
	m_sampg->begin();
	while(!m_sampg->end() ) {
	
		const sdb::LodSampleCache * cell = m_sampg->value();
		
		const int & nv = cell->numSamplesAtLevel(dlevel);
		
		if(nv>0) {
			const sdb::SampleCache * sps = cell->samplesAtLevel(dlevel);
			drs.draw(sps->points(), sps->normals(), nv);
		}
		
		m_sampg->next();
	}
	
	drs.end();
	
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
	m_fieldDrawer->drawEdge(m_mesher->coarseField() );
	m_fieldDrawer->drawNode(m_mesher->coarseField() );
	
	m_fieldDrawer->setDrawNodeSize(.1f);
	//m_fieldDrawer->drawEdge(m_mesher->fineField(3) );
	//m_fieldDrawer->drawEdge(m_mesher->fineField(4) );
	//m_fieldDrawer->drawNode(m_mesher->fineField(7) );
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
	