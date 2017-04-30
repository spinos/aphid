#include <QtGui>
#include <QtOpenGL>
#include "widget.h"
#include <GeoDrawer.h>
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
#include <topo/TriangleMeshCluster.h>
#include "../cactus.h"
#include <sdb/ebp.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{ 
	usePerspCamera(); 
	m_space.translate(1,1,1);
	m_roth = new RotationHandle(&m_space);
	m_triangles = new sdb::VectorArray<cvx::Triangle>();
/// prepare kd tree
	BoundingBox gridBox;
	KdEngine eng;
	eng.buildSource<cvx::Triangle, 3 >(m_triangles, gridBox,
									sCactusMeshVertices[0],
									sCactusNumTriangleIndices,
									sCactusMeshTriangleIndices);
									
	std::cout<<"\n kd tree source bbox"<<gridBox
			<<"\n n tri "<<m_triangles->size();
	
	TreeProperty::BuildProfile bf;
	bf._maxLeafPrims = 16;
	
	m_tree = new TreeTyp;
	
	eng.buildTree<cvx::Triangle, KdNode4, 4>(m_tree, m_triangles, gridBox, &bf);
	
typedef IntersectEngine<cvx::Triangle, KdNode4 > FIntersectTyp;

	FIntersectTyp ineng(m_tree);
	const float sz0 = m_tree->getBBox().getLongestDistance() * .89f;
	m_grid = new EbpGrid;
	m_grid->fillBox(m_tree->getBBox(), sz0 );
	m_grid->subdivideToLevel<FIntersectTyp>(ineng, 0, 3);
	m_grid->insertNodeAtLevel(3);
	m_grid->cachePositions();
	int np = m_grid->numParticles();
	qDebug()<<"\n grid n cell "<<m_grid->numCellsAtLevel(3);
	
	for(int i=0;i<20;++i) {
		m_grid->update();    
	}
	
	createParticles(np);
	
	TriangleMeshCluster * cluster = new TriangleMeshCluster;
	cluster->create(sCactusMeshVertices[0],
									sCactusNumTriangleIndices,
									sCactusMeshTriangleIndices);
		
	const int ns = cluster->numSites();
	std::cout<<"\n n site "<<ns;
	
	m_cs.reset(new int[ns]);
	
typedef ClosestToPointEngine<cvx::Triangle, KdNode4 > FClosestTyp;
	
	FClosestTyp clseng(m_tree);
	Vector3F top;
	
	const Vector3F * poss = m_grid->positions(); 

	int igeom, icomp;
	float contrib[4];	

    for(int i=0;i<np;++i) {
		clseng.closestTo(top, poss[i]);

		clseng.getGeomCompContribute(igeom, icomp, contrib);
#if 0
		std::cout<<"\n closest to geom_"<<igeom
					<<" comp_"<<icomp
					<<" contrib "<<contrib[0]<<","<<contrib[1]<<","<<contrib[2];
#endif

		cluster->assignToCliqueOrigin(&sCactusMeshTriangleIndices[icomp * 3], 
								i);
		
// column-major element[3] is translate  
		Float4 * pr = particleR(i);
            pr[0] = Float4(.25 ,0,0,top.x);
            pr[1] = Float4(0,.25 ,0,top.y);
            pr[2] = Float4(0,0,.25 ,top.z);
    }
	
	permutateParticleColors();
	
	float cliqueDistance = sz0 / 5.f;
	cluster->buildCliques(cliqueDistance);
	cluster->extractCliques(m_cs.get() );
	delete cluster;
	std::cout.flush();	
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{
	initGlsl();
}

void GLWidget::clientDraw()
{
	getDrawer()->setColor(0.f, .35f, .45f);
	
	getDrawer()->m_wireProfile.apply();
/*
	glEnableClientState(GL_VERTEX_ARRAY);
	
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)sCactusMeshVertices[0] );
	glDrawElements(GL_TRIANGLES, sCactusNumTriangleIndices, GL_UNSIGNED_INT, sCactusMeshTriangleIndices );
	
	glDisableClientState(GL_VERTEX_ARRAY);
*/
		
	glBegin(GL_TRIANGLES);
	for(int i=0;i<m_triangles->size();++i) {
		const cvx::Triangle * t = m_triangles->get(i);
		glVertex3fv((const GLfloat *)&t->P(0));
		glVertex3fv((const GLfloat *)&t->P(1));
		glVertex3fv((const GLfloat *)&t->P(2));
	}
	glEnd();
	
#if 0
	getDrawer()->setColor(0.f, .15f, .25f);

	DrawKdTree<cvx::Triangle, KdNode4 > drt(m_tree);
	drt.drawCells();
	
	getDrawer()->setColor(.35f, .15f, 0.f);
	
	DrawGrid<EbpGrid> dgd(m_grid);
	dgd.drawLevelCells(3);
#endif
	
	//getDrawer()->m_surfaceProfile.apply();
	getDrawer()->m_markerProfile.apply();
	
	drawParticles();
	
	drawFaceCliques();
}

void GLWidget::drawFaceCliques()
{
glColor3f(1,1,0);
	const int ns = sCactusNumTriangleIndices / 3;
	const float * pos = sCactusMeshVertices[0];
	glBegin(GL_TRIANGLES);
	for(int i=0;i<ns;++i) {
	
		const int c = m_cs[i]; 
		if(c < 0) {
			continue;
		}
		
		const Float4 * col = particleColor(c);
		glColor3f(col->x, col->y, col->z);
		
		const int * tri = &sCactusMeshTriangleIndices[i*3];
		
		glVertex3fv((const float *)&pos[tri[0] * 3]);
		glVertex3fv((const float *)&pos[tri[1] * 3]);
		glVertex3fv((const float *)&pos[tri[2] * 3]);
		
	}
	glEnd();
	
}

void GLWidget::clientSelect(QMouseEvent *event)
{
	m_roth->begin(getIncidentRay() );
	update();
}

void GLWidget::clientDeselect(QMouseEvent *event)
{
	m_roth->end();
	update();
}

void GLWidget::clientMouseInput(QMouseEvent *event)
{
	m_roth->rotate(getIncidentRay() );
	update();
}
	