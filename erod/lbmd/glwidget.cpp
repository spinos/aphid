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

using namespace aphid;

static const int NumPart = 3;
static const float PartP[3][3] = {
{11.33f, 22.43f, 4.78f},
{15.33f, 19.43f, 7.78f},
{17.33f, 13.23f, 10.08f},
};

static const float PartV[3][3] = {
{-1.33f, 2.43f, 0.78f},
{.73f, 1.43f, .38f},
{.83f, 1.23f, .08f},
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
	param._blockSize = 1.68f;
	m_latman->resetLattice(param);
	m_latman->injectParticles(PartP[0], PartV[0], NumPart);
	
}

GLWidget::~GLWidget()
{
}

void GLWidget::clientInit()
{
}

void GLWidget::clientDraw()
{
	glColor3f(.7f, .6f, .6f);
	for(int i=0;i<NumPart;++i) {
		glTranslatef(PartP[i][0], PartP[i][1], PartP[i][2]);
		getDrawer()->sphere(.5f);
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
	
	glColor3f(0.f, .4f, .2f);
	sdb::WorldGrid2<lbm::LatticeBlock >& grd = m_latman->grid();
	
	BoundingBox bbx;
	grd.begin();
	while(!grd.end() ) {
	
		bbx = grd.coordToCellBBox(grd.key() );
		
		getDrawer()->boundingBox(bbx);
		
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
					20.f, 44.f, 49.64101616f, 1.f};
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
