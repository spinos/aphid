#include <GeoDrawer.h>
#include <QtGui>
#include <QtOpenGL>
#include "widget.h"
#include <smp/EbpSphere.h>
#include <smp/SampleFilter.h>
#include <smp/EbpMeshSample.h>
#include <geom/DiscMesh.h>
#include <ogl/GlslInstancer.h>

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
	
	DiscMesh dsk(24);
	const int ndskv = dsk.numPoints();
	for(int i=0;i<ndskv;++i) {
		Vector3F vi = dsk.points()[i];
		vi.set(vi.x * 10.f, RandomFn11(), vi.y * 10.f);
		dsk.points()[i] = vi;
	}
	
	std::cout<<"\n disc bbx"<<dsk.calculateGeomBBox();
	
	EbpMeshSample smsh;
	smsh.sample(&dsk);
	
	m_flt = new smp::SampleFilter;
#if 0
	//m_flt->setPortion(.9f);
	//m_flt->setFacing(Vector3F(0.f, .5f, .5f) );
	m_flt->setAngle(.5f);
	m_flt->processFilter<EbpSphere>(m_grid);
#else
	m_flt->setPortion(.49f);
	m_flt->processFilter<EbpMeshSample>(&smsh);
	qDebug()<<"\n n disc samples "<<m_flt->numFilteredSamples();
#endif

    m_inst = new GlslInstancer;
	
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{
    //initGlsl();
	
    std::string diaglog;
    m_inst->diagnose(diaglog);
    std::cout<<diaglog;
    if(!m_inst->hasShaders() ) {
        m_inst->initializeShaders(diaglog);
    }
    std::cout<<diaglog;
    std::cout.flush();
    //return m_inst->isDiagnosed();
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

