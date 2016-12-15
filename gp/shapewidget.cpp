#include <GeoDrawer.h>
#include <GlslInstancer.h>
#include <QtGui>
#include <QtOpenGL>
#include <geom/SuperQuadricGlyph.h>
#include "shapewidget.h"
#include "RbfKernel.h"
#include "Covariance.h"
#include <Plane.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{
	m_instancer = new GlslLegacyInstancer;
	
#define NSamples 5
	const float SamplesAt[NSamples] = {.25f, .5f, 1.f, 1.5f, 2.f};

	m_numParticles = NSamples * NSamples;
	qDebug()<<" num examples "<<m_numParticles;
	m_particles.reset(new Float4[(m_numParticles+1) * 4]);
	m_glyphs.reset(new GlyphPtrType[m_numParticles+1]);
	
	int i, j, k, m;
	for(j=0;j<NSamples;++j) {
		for(i=0;i<NSamples;++i) {
			k = (j*NSamples + i)*4;
			
				m_particles[k  ] = Float4(2,0,0, 32.f* SamplesAt[i]);
				m_particles[k+1] = Float4(0,2,0, 32.f* SamplesAt[j]);
				m_particles[k+2] = Float4(0,0,2, 0);
				m_particles[k+3] = Float4(0,0,1,1);
				
				m_glyphs[k>>2] = new SuperQuadricGlyph(8);
				m_glyphs[k>>2]->computePositions(SamplesAt[i], SamplesAt[j]);
	
		}
	}
	
/// last one is predict			
	m_glyphs[m_numParticles] = new SuperQuadricGlyph(8);
	
	const int np = m_glyphs[0]->numPoints();
	const int ydim = np * 3;
	m_yMeasure = new DenseMatrix<float>(m_numParticles, ydim);
	for(j=0;j<NSamples;++j) {
		for(i=0;i<NSamples;++i) {
			k = j*NSamples + i;
			
			for(m=0;m<np;++m) {
				const Vector3F & v = m_glyphs[k]->points()[m];
				m_yMeasure->column(m         )[k] = v.x;
				m_yMeasure->column(m + np    )[k] = v.y;
				m_yMeasure->column(m + np * 2)[k] = v.z;
			}
		}
	}
	
	std::cout<<"\n y_train dim "<<m_yMeasure->numRows()
			<<"-by-"<<m_yMeasure->numCols();
	
	m_xMeasure = new DenseMatrix<float>(m_numParticles, 2);
	for(j=0;j<NSamples;++j) {
		for(i=0;i<NSamples;++i) {
			k = j*NSamples + i;
			m_xMeasure->column(0)[k] = SamplesAt[i];
			m_xMeasure->column(1)[k] = SamplesAt[j];
		}
	}
	
	m_rbf = new gpr::RbfKernel<float>(0.5f);
    m_covTrain = new gpr::Covariance<float, gpr::RbfKernel<float> >();
    m_covTrain->create(*m_xMeasure, *m_rbf);
	
	m_xPredict = new DenseMatrix<float>(1, 2);
	m_yPredict = new DenseMatrix<float>(1,ydim);
	
	predict(.98f, .98f);
	
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

	for(int i=0;i<=m_numParticles;++i) {
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

const DenseMatrix<float > & GLWidget::K() const
{ return m_covTrain->K(); }

void GLWidget::predict(const float & x0, const float & x1)
{
	qDebug()<<" predict at "<<x0<<" "<<x1;
	m_xPredict->column(0)[0] = x0;
	m_xPredict->column(1)[0] = x1;
	
	gpr::Covariance<float, gpr::RbfKernel<float> > covPredict;
	covPredict.create(*m_xPredict, *m_xMeasure, *m_rbf);
	
	DenseMatrix<float> KxKtraininv(covPredict.K().numRows(),
									m_covTrain->Kinv().numCols() );
									
	covPredict.K().mult(KxKtraininv, m_covTrain->Kinv() );
	
	KxKtraininv.mult(*m_yPredict, *m_yMeasure);
	
	const int np = m_glyphs[m_numParticles]->numPoints();
	Vector3F * ppredict = m_glyphs[m_numParticles]->points();
	for(int i=0;i<np;++i) {
		ppredict[i].x = m_yPredict->column(i         )[0];
		ppredict[i].y = m_yPredict->column(i + np    )[0];
		ppredict[i].z = m_yPredict->column(i + np * 2)[0];
	}
	
	m_particles[m_numParticles*4  ] = Float4(2,0,0,x0 * 32.f);
	m_particles[m_numParticles*4+1] = Float4(0,2,0,x1 * 32.f);
	m_particles[m_numParticles*4+2] = Float4(0,0,2, 1.f);
	m_particles[m_numParticles*4+3] = Float4(0,0,1,1);
	
}

void GLWidget::clientMouseInput(QMouseEvent *event)
{
	setUpdatesEnabled(false);
	Plane planexy(Vector3F::ZAxis, Vector3F(0.f, 0.f, 0.f) );
	Vector3F hit;
	float t;
	planexy.rayIntersect(*getIncidentRay(), hit, t, true);
	
	hit /= 32.f;
	
	if(hit.x < 1e-2f) hit.x = 1e-2f;
	if(hit.y < 1e-2f) hit.y = 1e-2f;
	if(hit.x > 2.2f) hit.x = 2.2f;
	if(hit.y > 2.2f) hit.y = 2.2f;
	
	predict(hit.x, hit.y);
	
	setUpdatesEnabled(true);
	update();
}
