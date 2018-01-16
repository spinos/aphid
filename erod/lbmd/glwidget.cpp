/*
 *  lbm collision
 */
#include <QtGui>
#include <QtOpenGL>
#include <math/BaseCamera.h>
#include "glwidget.h"
#include <GeoDrawer.h>
#include <math/Quaternion.h>
#include <lbm/LatticeManager.h>
#include <lbm/LatticeBlock.h>
#include <lbm/D3Q19DF.h>

using namespace aphid;

static const int NumPart = 9;
static const float PartP[9][3] = {
{12.033f, 19.9743f, 7.978f},
{10.33f, 18.43f, 9.9178f},
{12.33f, 17.23f, 10.08f},
{10.143f, 15.93f, 9.78f},
{11.033f, 14.023f, 11.08f},
{10.93f, 13.023f, 10.438f},
{12.33f, 15.923f, 9.6708f},
{12.17543f, 14.9143f, 12.708f},
{11.943f, 12.53f, 13.18f},
};

static const float PartV[9][3] = {
{.33f - 1.f, -.843f - 1.f, 1.978f},
{.73f + 1.f, .53f - 1.f, -.7638f},
{.383f + 1.f, -.23f - 1.f, -1.98f},
{.53f + 1.f, -.83f - 1.f, 1.828f},
{.73f + 1.f, .973f + 1.f, 1.7408f},
{.283f + 1.f, -.123f + 1.f, 1.358f},
{.143f + 1.f, .523f - 1.f, -1.508f},
{.2041f - 1.f, .643f - 1.f, -1.3138f},
{.03f + 1.f, .95606f + 1.f, -1.9467f},
};

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	perspCamera()->setFarClipPlane(2000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(2000.f);
	orthoCamera()->setNearClipPlane(1.f);
	usePerspCamera();
	resetView();
	m_latman = new lbm::LatticeManager;
	lbm::LatticeParam param;
	param._blockSize = 23.f;
	m_latman->resetLattice(param);
	m_latman->injectParticles(PartP[0], PartV[0], NumPart);
	m_latman->finishInjectingParticles();
	m_latman->simulationStep();
	
	m_nodeCenter = new DenseVector<float>(lbm::LatticeBlock::BlockLength * 3);
	m_nodeU = new DenseVector<float>(lbm::LatticeBlock::BlockLength * 3);
	
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
	
	float u[3] = {.5f, .2f, -1.1f};
	std::cout<<"\n discretize u "<<u[0]<<","<<u[1]<<","<<u[2];
	
	lbm::D3Q19DF::DiscretizeVelocity(f_i, u, 0);
	
	std::cout<<"\n f_i";
	for(int i=0;i<19;++i) {
		std::cout<<"\n f_"<<i<<" "<<f_i[i][0];
	}
	
	float rho;
	lbm::D3Q19DF::CompressibleVelocity(u, rho, f_i, 0);
	
	std::cout<<"\n compose u "<<u[0]<<","<<u[1]<<","<<u[2]
		<<"\n rho "<<rho;
		
	lbm::D3Q19DF::IncompressibleVelocity(u, rho, f_i, 0);
	
	std::cout<<"\n incompose u "<<u[0]<<","<<u[1]<<","<<u[2]
		<<"\n rho "<<rho;

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
	glColor3f(.9f, .7f, .0f);
#if 1
	for(int i=0;i<NumPart;++i) {
		glTranslatef(PartP[i][0], PartP[i][1], PartP[i][2]);
		getDrawer()->sphere(.0625f);
		glTranslatef(-PartP[i][0], -PartP[i][1], -PartP[i][2]);
		
	}

	glBegin(GL_LINES);
	for(int i=0;i<NumPart;++i) {
		glVertex3fv(PartP[i]);
		glVertex3f(PartP[i][0] + PartV[i][0], 
					PartP[i][1] + PartV[i][1], 
					PartP[i][2] + PartV[i][2]);
		
	}
	glEnd();
#endif	
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

	blk->extractCellVelocities(m_nodeU->v() );
	
	glBegin(GL_LINES);
	for(int i=0;i<lbm::LatticeBlock::BlockLength;++i) {
		glVertex3fv(&m_nodeCenter->c_v()[i*3]);			
		glVertex3f(m_nodeCenter->c_v()[i*3] + m_nodeU->c_v()[i*3] * lbm::LatticeBlock::CellSize,
				m_nodeCenter->c_v()[i*3 + 1] + m_nodeU->c_v()[i*3 + 1] * lbm::LatticeBlock::CellSize,
				m_nodeCenter->c_v()[i*3 + 2] + m_nodeU->c_v()[i*3 + 2] * lbm::LatticeBlock::CellSize);
	}
	
	glEnd();
	
	glColor3f(0, 1, 0);
	
	float u[3];
	
	glBegin(GL_LINES);
	for(int i=0;i<NumPart;++i) {
		blk->evaluateVelocityAtPosition(u, PartP[i]);
		
		glVertex3fv(PartP[i]);	
		glVertex3f(PartP[i][0] + u[0] * lbm::LatticeBlock::CellSize,
				PartP[i][1] + u[1] * lbm::LatticeBlock::CellSize,
				PartP[i][2] + u[2] * lbm::LatticeBlock::CellSize);
	}
	
	glEnd();	
}
