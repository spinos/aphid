#include <QtGui>
#include <QtOpenGL>
#include <GeodesicSphereMesh.h>
#include "instwidget.h"
#include <GeoDrawer.h>
#include <GlslInstancer.h>
#include "ebp.h"

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{
	m_sphere = new TriangleGeodesicSphere(7);
    m_instancer = new GlslLegacyInstancer;
	
	std::vector<cvx::Triangle * > tris;
	cvx::Triangle * ta = new cvx::Triangle;
	ta->set(Vector3F(-8, -3, 8), Vector3F(8, 0, 1), Vector3F(-1, 12, -4) );
	tris.push_back(ta);
	cvx::Triangle * tb = new cvx::Triangle;
	tb->set(Vector3F(-1, 12, -4), Vector3F(8, 0, 1), Vector3F(10, 7, -13) );
	tris.push_back(tb);
	
	sdb::Sequence<int> sels;
	sels.insert(0);
	sels.insert(1);
	
typedef PrimInd<sdb::Sequence<int>, std::vector<cvx::Triangle * >, cvx::Triangle > TIntersect;
	TIntersect fintersect(&sels, &tris);
	
	float gz = 2.1f;
	EbpGrid grid;
	grid.fillBox(fintersect, 16.8);
	grid.subdivideToLevel<TIntersect>(fintersect, 0, 3);
	grid.insertNodeAtLevel(3);
	std::cout<<"\n grid n cell "<<grid.numCellsAtLevel(3);
	
	m_numParticles = grid.countNodes();
	qDebug()<<" num instances "<<m_numParticles;
	
	Vector3F * poss = new Vector3F[m_numParticles]; 
	grid.extractPos(poss, m_numParticles);
    
    m_particles = new Float4[m_numParticles * 4];
    
// column-major element[3] is translate  
    for(int i=0;i<m_numParticles;++i) {
		int k = i*4;
            m_particles[k] = Float4(1 ,0,0,poss[i].x);
            m_particles[k+1] = Float4(0,1 ,0,poss[i].y);
            m_particles[k+2] = Float4(0,0,1 ,poss[i].z);
            m_particles[k+3] = Float4(0,0,1,1);
    }
	
	delete[] poss;

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
	
	for(int i=0;i<m_numParticles;++i) {
	    const Float4 *d = &m_particles[i*4];
	    glMultiTexCoord4fv(GL_TEXTURE1, (const float *)d);
	    glMultiTexCoord4fv(GL_TEXTURE2, (const float *)&d[1]);
	    glMultiTexCoord4fv(GL_TEXTURE3, (const float *)&d[2]);
	    
	    glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);
        glNormalPointer(GL_FLOAT, 0, (GLfloat*)m_sphere->vertexNormals());
        glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)m_sphere->points());
        glDrawElements(GL_TRIANGLES, m_sphere->numIndices(), GL_UNSIGNED_INT, m_sphere->indices());
        
        glDisableClientState(GL_NORMAL_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
	}
	
	m_instancer->programEnd();
	
}


