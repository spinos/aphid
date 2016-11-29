#include <QtGui>
#include <QtOpenGL>
#include <GeodesicSphereMesh.h>
#include <geom/SuperQuadricGlyph.h>
#include "instwidget.h"
#include <GeoDrawer.h>
#include <GlslInstancer.h>
#include "ebp.h"

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{
	m_glyph = new SuperQuadricGlyph(8);
	m_glyph->computePositions(2.5f, .5f);
	m_sphere = new TriangleGeodesicSphere(7);
    m_instancer = new GlslLegacyInstancer;
	
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
	int numParticles = m_grid->numParticles();
	qDebug()<<" num instances "<<numParticles;
	
	m_grid->update();
	
	for(int i=0;i<20;++i) {
		m_grid->update();    
	}
	
    m_particles = new Float4[numParticles * 4];
	const Vector3F * poss = m_grid->positions(); 
	
// column-major element[3] is translate  
    for(int i=0;i<numParticles;++i) {
		int k = i*4;
            m_particles[k] = Float4(1 ,0,0,poss[i].x);
            m_particles[k+1] = Float4(0,1 ,0,poss[i].y);
            m_particles[k+2] = Float4(0,0,1 ,poss[i].z);
            m_particles[k+3] = Float4(0,0,1,1);
    }
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{
    std::string diaglog;
    m_instancer->diagnose(diaglog);
    std::cout<<diaglog;
    m_instancer->initializeShaders(diaglog);
    std::cout<<diaglog;
    std::cout.flush();
    
}

void GLWidget::clientDraw()
{
    getDrawer()->m_markerProfile.apply();
	getDrawer()->setColor(.75f, .85f, .7f);

	m_instancer->programBegin();
#if 0
	m_particles[0].set(10.f, 0.f, 0.f, 0.f);	
	m_particles[1].set(0.f, 10.f, 0.f, 0.f);	
	m_particles[2].set(0.f, 0.f, 10.f, 0.f);	
	int numParticles = 1;
#else
	int numParticles = m_grid->numParticles();
#endif
	for(int i=0;i<numParticles;++i) {
	    const Float4 *d = &m_particles[i*4];
	    glMultiTexCoord4fv(GL_TEXTURE1, (const float *)d);
	    glMultiTexCoord4fv(GL_TEXTURE2, (const float *)&d[1]);
	    glMultiTexCoord4fv(GL_TEXTURE3, (const float *)&d[2]);
	    
	    glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);
        glNormalPointer(GL_FLOAT, 0, (GLfloat*)m_glyph->vertexNormals());
        glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)m_glyph->points());
        glDrawElements(GL_TRIANGLES, m_glyph->numIndices(), GL_UNSIGNED_INT, m_glyph->indices());
        
        glDisableClientState(GL_NORMAL_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
	}
	
	m_instancer->programEnd();
	
#if 0
	for(int i=0;i<20;++i) {
		m_grid->update();    
	}
	
	const Vector3F * poss = m_grid->positions(); 
	
    for(int i=0;i<numParticles;++i) {
		int k = i*4;
            m_particles[k].w = poss[i].x;
            m_particles[k+1].w = poss[i].y;
            m_particles[k+2].w = poss[i].z;
    }
#endif
}
