#include <GeoDrawer.h>
#include <QtGui>
#include <QtOpenGL>
#include "widget.h"
#include <smp/EbpSphere.h>
#include <smp/SampleFilter.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{
	updateGlyph(.5f, .5f);
	
	m_grid = new EbpSphere;
	
	const int np = m_grid->numParticles();
	qDebug()<<"\n n cell "<<m_grid->numSamples()
			<<" num instances "<<np;
	
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
	
	m_flt = new smp::SampleFilter;
	//m_flt->setPortion(.9f);
	//m_flt->setFacing(Vector3F(0.f, .5f, .5f) );
	m_flt->setAngle(.5f);
	m_flt->processFilter<EbpSphere>(m_grid);
	
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
	
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)m_flt->filteredSamples() );
	glDrawArrays(GL_POINTS, 0, m_flt->numFilteredSamples());
	
	glDisableClientState(GL_VERTEX_ARRAY);
}

