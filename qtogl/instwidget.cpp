#include <QtGui>
#include <QtOpenGL>
#include <GeodesicSphereMesh.h>
#include "instwidget.h"
#include <GeoDrawer.h>
#include <GlslInstancer.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{
    m_sphere = new TriangleGeodesicSphere(7);
    m_instancer = new GlslLegacyInstancer;
    const int N = 25;
    m_numParticles = N * N;
    m_particles = new Float4[m_numParticles * 4];
    qDebug()<<" num instances "<<m_numParticles;
// column-major element[3] is translate  
    for(int j=0;j<N;++j) {
        for(int i=0;i<N;++i) {
            int k = (j*N+i)*4;
            m_particles[k] = Float4(1+RandomF01(),0,0,i*7);
            m_particles[k+1] = Float4(0,1+RandomF01(),0,0.1*(N-j));
            m_particles[k+2] = Float4(0,0,1+RandomF01(),(j-N)*7);
            m_particles[k+3] = Float4(0,0,1,1);
        }
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


