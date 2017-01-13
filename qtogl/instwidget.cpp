#include <QtGui>
#include <QtOpenGL>
#include "instwidget.h"
#include <GeoDrawer.h>
#include <sdb/ebp.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{
	updateGlyph(.5f, .5f);
	
	std::vector<cvx::Triangle * > tris;
	cvx::Triangle * ta = new cvx::Triangle;
	ta->set(Vector3F(-12, -2, 8), Vector3F(8, 0, 1), Vector3F(-1, 12, -4) );
	tris.push_back(ta);
	cvx::Triangle * tb = new cvx::Triangle;
	tb->set(Vector3F(-1, 12, -4), Vector3F(8, 0, 1), Vector3F(5, 0, -20) );
	tris.push_back(tb);
	
	sdb::Sequence<int> sels;
	sels.insert(0);
	sels.insert(1);
	
typedef PrimInd<sdb::Sequence<int>, std::vector<cvx::Triangle * >, cvx::Triangle > TIntersect;
	TIntersect fintersect(&sels, &tris);
	
	float gz = 1.5f;
	m_grid = new EbpGrid;
	m_grid->fillBox(fintersect, 12);
	m_grid->subdivideToLevel<TIntersect>(fintersect, 0, 3);
	m_grid->insertNodeAtLevel(3);
	m_grid->cachePositions();
	const int np = m_grid->numParticles();
	qDebug()<<"\n n cell "<<m_grid->numCellsAtLevel(3)
			<<" num instances "<<np;
	
	for(int i=0;i<20;++i) {
		m_grid->update();    
	}
	
	createParticles(np);
	
    const Vector3F * poss = m_grid->positions(); 
	
// column-major element[3] is translate  
    for(int i=0;i<np;++i) {
		Float4 * pr = particleR(i);
            pr[0] = Float4(1 ,0,0,poss[i].x);
            pr[1] = Float4(0,1 ,0,poss[i].y);
            pr[2] = Float4(0,0,1 ,poss[i].z);
    }
	
	permutateParticleColors();
	
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{
    initGlsl();
}

void GLWidget::clientDraw()
{
    getDrawer()->m_markerProfile.apply();
	getDrawer()->setColor(.5f, .85f, .7f);
	
	drawParticles();
	
#if 0
	for(int i=0;i<20;++i) {
		m_grid->update();    
	}
	
	const Vector3F * poss = m_grid->positions(); 
	
    for(int i=0;i<numParticles();++i) {
		Float4 * pr = particleR(i);
            pr[0].w = poss[i].x;
            pr[1].w = poss[i].y;
            pr[2].w = poss[i].z;
    }
#endif
}
