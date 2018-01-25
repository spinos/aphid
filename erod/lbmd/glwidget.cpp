/*
 *  lbm collision
 */
#include <QtGui>
#include <QtOpenGL>
#include <math/BaseCamera.h>
#include "glwidget.h"
#include <GeoDrawer.h>
#include <math/Quaternion.h>
#include <lbm/VolumeResponse.h>
#include <lbm/LatticeBlock.h>
#include <lbm/D3Q19DF.h>
#include <math/miscfuncs.h>

using namespace aphid;

static const int NumPart = 4096;

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	perspCamera()->setFarClipPlane(2000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(2000.f);
	orthoCamera()->setNearClipPlane(1.f);
	usePerspCamera();
	resetView();
	
	m_particleX = new DenseVector<float>(NumPart * 3);
	m_particleU = new DenseVector<float>(NumPart * 3);
	m_particleUhat = new DenseVector<float>(NumPart * 3);
	
	for(int i=0;i<(NumPart>>1);++i) {
	
		m_particleX->v()[i * 3] = 7.f + RandomFn11() * 4.f;
		m_particleX->v()[i * 3 + 1] = 12.f + RandomFn11() * 4.f;
		m_particleX->v()[i * 3 + 2] = 9.f + RandomFn11() * 5.f;
		
		m_particleU->v()[i * 3] = 2.5;
		m_particleU->v()[i * 3 + 1] = RandomFn11() * .5f;
		m_particleU->v()[i * 3 + 2] = .5 - RandomF01() * .5f;
	
	}
	
	for(int i=(NumPart>>1);i<NumPart;++i) {
	
		m_particleX->v()[i * 3] = 25.f + RandomFn11() * 4.f;
		m_particleX->v()[i * 3 + 1] = 12.f + RandomFn11() * 5.f;
		m_particleX->v()[i * 3 + 2] = 9.f + RandomFn11() * 4.f;
		
		m_particleU->v()[i * 3] = -2.5;
		m_particleU->v()[i * 3 + 1] = RandomFn11() * .5f;
		m_particleU->v()[i * 3 + 2] = .5 + RandomF01() * .5f;
	
	}
	memcpy(m_particleUhat->v(), m_particleU->v(), NumPart * 12);

	
	m_latman = new lbm::VolumeResponse;
	lbm::LatticeParam param;
	param._blockSize = 32.f;
	param._inScale = .1f;
	param._outScale = 15.f;
	m_latman->setParam(param);
	m_latman->solveParticles(m_particleU->v(), m_particleX->v(), NumPart);
	
	m_nodeCenter = new DenseVector<float>(lbm::LatticeBlock::BlockLength * 3);
	m_nodeU = new DenseVector<float>(lbm::LatticeBlock::BlockLength * 3);
	m_nodeRho = new DenseVector<float>(lbm::LatticeBlock::BlockLength);
	
	std::cout<<"\n e_i";
	for(int i=0;i<19;++i) {
		std::cout<<"\n "<<lbm::D3Q19DF::e_alpha[0][i]
					<<" "<<lbm::D3Q19DF::e_alpha[1][i]
					<<" "<<lbm::D3Q19DF::e_alpha[2][i]
					<<" inv "<<lbm::D3Q19DF::inv_e_alpha[0][i]
					<<" "<<lbm::D3Q19DF::inv_e_alpha[1][i]
					<<" "<<lbm::D3Q19DF::inv_e_alpha[2][i];
	}
	
	float* f_i[19];
	for(int i=0;i<19;++i) {
		f_i[i] = new float;
	}
	
	for(int i=0;i<19;++i) {
		lbm::D3Q19DF::SetWi(f_i[i], 1, i);
	}
	
	std::cout<<"\n w_i";
	for(int i=0;i<19;++i) {
		std::cout<<"\n f_"<<i<<" "<<f_i[i][0];
	}
	
	float u[3] = {0,0,-0.97335};
	std::cout<<"\n discretize u "<<u[0]<<","<<u[1]<<","<<u[2];
	
	lbm::D3Q19DF::DiscretizeVelocity(f_i, 1.f, u, 0);
	
	std::cout<<"\n f_i";
	for(int i=0;i<19;++i) {
		std::cout<<"\n f_"<<i<<" "<<f_i[i][0];
	}
	
	float rho;		
	lbm::D3Q19DF::IncompressibleVelocity(u, rho, f_i, 0);
	
	std::cout<<"\n incompressible u "<<u[0]<<","<<u[1]<<","<<u[2]
		<<"\n rho "<<rho;

	lbm::D3Q19DF::CompressibleVelocity(u, rho, f_i, 0);
	
	std::cout<<"\n compressible u "<<u[0]<<","<<u[1]<<","<<u[2]
		<<"\n rho "<<rho;
		
	f_i[2][0] = 0.0277778;
	f_i[13][0] = -0.0444444;
	f_i[15][0] = 0.025;
	
	float uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
	lbm::D3Q19DF::Relaxing(f_i, u, uu, rho, 1.5f, 0);
	
	lbm::D3Q19DF::CompressibleVelocity(u, rho, f_i, 0);
	
	std::cout<<"\n relaxed\n compressible u "<<u[0]<<","<<u[1]<<","<<u[2]
		<<"\n rho "<<rho;
		
	std::cout<<"\n cell size "<<lbm::LatticeBlock::CellSize;
		
	std::cout.flush();
	
}

GLWidget::~GLWidget()
{
}

void GLWidget::clientInit()
{
}

void GLWidget::clientDraw()
{	
	glColor3f(1, 1, 0);
	const float vscale = .125f;

	glBegin(GL_LINES);
	for(int i=0;i<NumPart;++i) {

		const float* xi = &m_particleX->c_v()[i * 3];
		glVertex3fv(xi);	
		const float* ui = &m_particleUhat->c_v()[i * 3];
		glVertex3f(xi[0] + ui[0] * vscale,
				xi[1] + ui[1] * vscale,
				xi[2] + ui[2] * vscale);
	}
	
	glEnd();

	glColor3f(0, 1, 0);

	glBegin(GL_LINES);
	for(int i=0;i<NumPart;++i) {

		const float* xi = &m_particleX->c_v()[i * 3];
		glVertex3fv(xi);	
		const float* ui = &m_particleU->c_v()[i * 3];
		glVertex3f(xi[0] + ui[0] * vscale,
				xi[1] + ui[1] * vscale,
				xi[2] + ui[2] * vscale);
	}
	
	glEnd();
	
	sdb::WorldGrid2<lbm::LatticeBlock >& grd = m_latman->grid();
	
	BoundingBox bbx;
	
	glColor3f(0.f, .1f, .1f);
		
	grd.begin();
	while(!grd.end() ) {
	
		bbx = grd.coordToCellBBox(grd.key() );
		
		getDrawer()->boundingBox(bbx);
		
		drawBlock(grd.value() );
		
		grd.next();
	}
	
	
}
//! [7]

//! [9]
void GLWidget::clientSelect(Vector3F & origin, Vector3F & ray, Vector3F & hit)
{
}
//! [9]

void GLWidget::clientDeselect()
{
}

//! [10]
void GLWidget::clientMouseInput(Vector3F & stir)
{
}
//! [10]

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	
	Base3DView::keyPressEvent(e);
}

void GLWidget::keyReleaseEvent(QKeyEvent *event)
{
	switch (event->key()) {
		case Qt::Key_A:
			simulationStep(true);
			break;
		case Qt::Key_Space:
			simulationStep(false);
			break;
		default:
			break;
	}
	
	Base3DView::keyReleaseEvent(event);
}

void GLWidget::resetPerspViewTransform()
{
static const float mm[16] = {1.f, 0.f, 0.f, 0.f,
					0.f, 0.8660254f, -0.5f, 0.f,
					0.f, 0.5f, 0.8660254f, 0.f,
					20.f, 40.f, 49.64101616f, 1.f};
	Matrix44F mat(mm);
	perspCamera()->setViewTransform(mat, 40.f);
}

void GLWidget::resetOrthoViewTransform()
{
static const float mm1[16] = {1.f, 0.f, 0.f, 0.f,
					0.f, 0.8660254f, -0.5f, 0.f,
					0.f, 0.5f, 0.8660254f, 0.f,
					2.f, 20.f, 34.64101616f, 1.f};
	Matrix44F mat(mm1);
	orthoCamera()->setViewTransform(mat, 15.f);
	orthoCamera()->setHorizontalAperture(15.f);
}

void GLWidget::drawBlock(aphid::lbm::LatticeBlock* blk)
{
	blk->extractCellCenters(m_nodeCenter->v() );
	blk->extractCellDensities(m_nodeRho->v() );
	blk->extractCellVelocities(m_nodeU->v() );
	const float scaling = lbm::LatticeBlock::CellSize * 3.35f;
	
	glColor3f(0.f, .1f, .1f);
	
	glBegin(GL_LINES);
	for(int i=0;i<lbm::LatticeBlock::BlockLength;++i) {
		glVertex3fv(&m_nodeCenter->c_v()[i*3]);			
		glVertex3f(m_nodeCenter->c_v()[i*3] + m_nodeU->c_v()[i*3]      * scaling,
				m_nodeCenter->c_v()[i*3 + 1] + m_nodeU->c_v()[i*3 + 1] * scaling,
				m_nodeCenter->c_v()[i*3 + 2] + m_nodeU->c_v()[i*3 + 2] * scaling);
	}
	glEnd();
	
	for(int i=0;i<lbm::LatticeBlock::BlockLength;++i) {
		
		const float* pv = &m_nodeCenter->c_v()[i*3];
		const float& prho = m_nodeRho->c_v()[i];
		
		if(prho > 1.03f) {
			glTranslatef(pv[0], pv[1], pv[2]);
			glColor3f(1,0,0);
			getDrawer()->sphere(.0625f);
			glTranslatef(-pv[0], -pv[1], -pv[2]);
		} else if(prho < .99f) {
			glTranslatef(pv[0], pv[1], pv[2]);
			glColor3f(0,0,1);
			getDrawer()->sphere(.0625f);
			glTranslatef(-pv[0], -pv[1], -pv[2]);
		}
		
	}
}

void GLWidget::simulationStep(bool toMoveParticles)
{
	if(toMoveParticles) {
		const float g = .098f;
		for(int i=0;i<NumPart;++i) {
			float* xi = &m_particleX->v()[i * 3];
			float* ui = &m_particleU->v()[i * 3];
			
			xi[0] += ui[0] * .04f;
			xi[1] += ui[1] * .04f;
			xi[2] += ui[2] * .04f;
			
			ui[0] += RandomFn11() * 0.14f;
			ui[1] += g;
			ui[2] += RandomFn11() * 0.14f;
			
		}
		
		m_particleUhat->copy(*m_particleU);
		
		m_latman->solveParticles(m_particleU->v(), m_particleX->v(), NumPart);
	} else {
		m_latman->simulationStep();
	}
	update();
}
