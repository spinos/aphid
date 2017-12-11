/*
 *  real instance
 *
 *  http://www.informit.com/articles/article.aspx?p=1377833&seqNum=8
 *  Vertex-Array Objects
 *  http://www.songho.ca/opengl/gl_vbo.html#create
 *  https://github.com/erwincoumans/experiments/blob/master/rendering/GLSL_Instancing/main.cpp
 *  glsl instancing
 *
 */

#include <GeoDrawer.h>
#include <QtGui>
#include <QtOpenGL>
#include "widget.h"
#include <smp/EbpSphere.h>
#include <smp/SampleFilter.h>
#include <smp/EbpMeshSample.h>
#include <geom/DiscMesh.h>
#include <ogl/GlslInstancer.h>

static GLuint               cube_vao;

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
    initInst();
}

void GLWidget::clientDraw()
{
    getDrawer()->m_markerProfile.apply();
	getDrawer()->setColor(1.f, 1.f, 1.f);
	
	//drawParticles();
	//drawSamples();
	drawTest();
}

void GLWidget::drawSamples()
{
    glEnableClientState(GL_VERTEX_ARRAY);
	
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)m_flt->filteredSamples() );
	glDrawArrays(GL_POINTS, 0, m_flt->numFilteredSamples());
	
	glDisableClientState(GL_VERTEX_ARRAY);
}


void GLWidget::initInst()
{
	GLfloat  cubeVerts[][3] = {
            { -1.0, -1.0, -1.0 },
            { -1.0, -1.0,  1.0 },
            { -1.0,  1.0, -1.0 },
            { -1.0,  1.0,  1.0 },
            {  1.0, -1.0, -1.0 },
            {  1.0, -1.0,  1.0 },
            {  1.0,  1.0, -1.0 },
            {  1.0,  1.0,  1.0 },
        };

        GLfloat  cubeColors[][3] = {
            {  0.0,  0.0,  0.0 },
            {  0.0,  0.0,  1.0 },
            {  0.0,  1.0,  0.0 },
            {  0.0,  1.0,  1.0 },
            {  1.0,  0.0,  0.0 },
            {  1.0,  0.0,  1.0 },
            {  1.0,  1.0,  0.0 },
            {  1.0,  1.0,  1.0 },
        };

        GLubyte  cubeIndices[] = {
            0, 1, 3, 2,
            4, 6, 7, 5,
            2, 3, 7, 6,
            0, 4, 5, 1,
            0, 2, 6, 4,
            1, 5, 7, 3
        };
        
        std::cout<<"\n size of cube vertices "<<sizeof(cubeVerts)
	<<"\n num of cube indices "<<sizeof(cubeIndices);
	std::cout.flush();
	
    glGenVertexArrays(1, &cube_vao);

	glBindVertexArray(cube_vao);
	
    GLuint cube_vbo;	
	glGenBuffers(1, &cube_vbo);
	
	glBindBuffer(GL_ARRAY_BUFFER, cube_vbo);   
	glBufferData(GL_ARRAY_BUFFER, 
			sizeof(cubeVerts), cubeVerts, 
			GL_STATIC_DRAW);
	glEnableClientState(GL_VERTEX_ARRAY); 
	glVertexPointer(3, GL_FLOAT, 0, 0);    
	
	GLuint index_vbo;
	glGenBuffers(1, &index_vbo);
	
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_vbo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER,
            sizeof(cubeIndices), cubeIndices, 
			GL_STATIC_DRAW);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER,0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	
}

void GLWidget::drawTest()
{
    glBindVertexArray(cube_vao);

    glDrawElements(GL_QUADS, 24, GL_UNSIGNED_BYTE,0);

    glBindVertexArray(0);

}
