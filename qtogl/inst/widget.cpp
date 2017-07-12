#include <GeoDrawer.h>
#include <QtGui>
#include <QtOpenGL>
#include "widget.h"
#include <sdb/ebp.h>
#include <geom/PrimInd.h>
#include <geom/GeodesicSphereMesh.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{
	updateGlyph(.5f, .5f);
	
	TriangleGeodesicSphere geodsph(6);
	const int nv = geodsph.numPoints();
	Vector3F * sphps = geodsph.points();
	for(int i=0;i<nv;++i) {
	    sphps[i] *= 9.f;
	}
	std::cout<<"\n geod sph bbox "<<geodsph.calculateGeomBBox();
	
	std::vector<cvx::Triangle * > tris;
    sdb::Sequence<int> sels;
    
	const int nt = geodsph.numTriangles();
	for(int i=0;i<nt;++i) {
	    const unsigned * trii = geodsph.triangleIndices(i);
	    cvx::Triangle * ta = new cvx::Triangle;
	    ta->set(sphps[trii[0]], sphps[trii[1]], sphps[trii[2]] );
	    tris.push_back(ta);
	    sels.insert(i);
	    
	}
		
typedef PrimInd<sdb::Sequence<int>, std::vector<cvx::Triangle * >, cvx::Triangle > TIntersect;
	TIntersect fintersect(&sels, &tris);
	
#define GrdLvl 3
	m_grid = new EbpGrid;
	m_grid->fillBox(fintersect, 9.3);
	m_grid->subdivideToLevel<TIntersect>(fintersect, 0, GrdLvl);
	m_grid->insertNodeAtLevel(GrdLvl);
	m_grid->cachePositions();
	const int np = m_grid->numParticles();
	qDebug()<<"\n n cell "<<m_grid->numCellsAtLevel(GrdLvl)
			<<" num instances "<<np;
	
	for(int i=0;i<10;++i) {
		m_grid->updateNormalized(10.f);    
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
	
	m_samples = new Vector3F[np];
	for(int i=0;i<np;++i) {
	    m_samples[i] = poss[i];
	}
	
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
	getDrawer()->setColor(1.f, 1.f, 1.f);
	
	//drawParticles();
	drawSamples();
	
}

void GLWidget::drawSamples()
{
    glEnableClientState(GL_VERTEX_ARRAY);
	
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)m_samples);
	glDrawArrays(GL_POINTS, 0, numParticles());
	
	glDisableClientState(GL_VERTEX_ARRAY);
}

