#include <QtGui>
#include <QtOpenGL>
#include <geom/SuperQuadricGlyph.h>
#include "shapewidget.h"
#include <GeoDrawer.h>
#include <GlslInstancer.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{
	m_instancer = new GlslLegacyInstancer;
	
#define NSamples 5
	const float SamplesAt[NSamples] = {.25f, .5f, 1.f, 1.5f, 2.f};

	m_numParticles = NSamples * NSamples;
	qDebug()<<" num examples "<<m_numParticles;
	m_particles = new Float4[m_numParticles * 4];
	
	m_glyphs.reset(new GlyphPtrType[m_numParticles]);
	
	for(int j=0;j<NSamples;++j) {
		for(int i=0;i<NSamples;++i) {
			int k = (j*NSamples + i)*4;
				m_particles[k] = Float4(2,0,0, 8.f*i + 2);
				m_particles[k+1] = Float4(0,2 ,0, 8.f*j + 2);
				m_particles[k+2] = Float4(0,0,2 , 0);
				m_particles[k+3] = Float4(0,0,1,1);
				
				m_glyphs[k>>2] = new SuperQuadricGlyph(8);
				m_glyphs[k>>2]->computePositions(SamplesAt[i], SamplesAt[j]);
	
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
		
		const GlyphPtrType aglyph = m_glyphs[i]; 
	    
	    glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);
        glNormalPointer(GL_FLOAT, 0, (GLfloat*)aglyph->vertexNormals());
        glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)aglyph->points());
        glDrawElements(GL_TRIANGLES, aglyph->numIndices(), GL_UNSIGNED_INT, aglyph->indices());
        
        glDisableClientState(GL_NORMAL_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
	}
	
	m_instancer->programEnd();
	
}
