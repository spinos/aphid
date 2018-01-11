/*
 *  shape matching region
 */
#include <QtGui>
#include <QtOpenGL>
#include <math/BaseCamera.h>
#include "glwidget.h"
#include <GeoDrawer.h>
#include <math/Quaternion.h>
#include <pbd/ShapeMatchingRegion.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	perspCamera()->setFarClipPlane(2000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(2000.f);
	orthoCamera()->setNearClipPlane(1.f);
	usePerspCamera();
	resetView();
	
	static const float sX[6][3] = {
	{1, 1, 0},
	{1, 1, 3},
	{4, 2, 3},
	{4, 2, 0},
	{7, 1, 0},
	{7, 1, 3},
	};

	static const float invMass[6] = {
	0.f, 1.f, 1.f, 1.f, 1.f, 1.f 
	};
	
	static const int vind[6] = {
	0, 1, 2, 3, 4, 5
	};
	
	static const int eind[14] = {
	0, 1, 
	1, 2, 
	2, 3, 
	3, 0, 
	3, 4, 
	4, 5,
	5, 2,
	};
	
	m_smr = new pbd::ShapeMatchingRegion;
	
	pbd::RegionVE prof;
	prof._nv = 6;
	prof._ne = 7;
	prof._vinds = new int[6];
	memcpy(prof._vinds, vind, 24);
	
	prof._einds = new int[14];
	memcpy(prof._einds, eind, 56);
	
	m_smr->createRegion(prof, sX[0], invMass);
	std::cout.flush();
	
	m_p = new float[18];
	memcpy(m_p, sX[0], 72);
	
	m_activeV = 5;
	
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
	int v1, v2;
	glBegin(GL_LINES);
	const int& ne = m_smr->numEdges();
	for(int i=0;i<ne;++i) {
		m_smr->getEdge(v1, v2, i);
		glVertex3fv(&m_p[v1 * 3]);
		glVertex3fv(&m_p[v2 * 3]);
	}
	glEnd();
	
	glColor3f(.1f, .99f, .1f);
	glBegin(GL_LINE_STRIP);
	for(int i=0;i<6;++i) {
		glVertex3fv(m_smr->goalPosition(i));
	}
	glEnd();
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
			m_p[m_activeV * 3] -= 0.4f;
			m_smr->updateRegion(m_p);
			break;
		case Qt::Key_D:
			m_p[m_activeV * 3] += 0.4f;
			m_smr->updateRegion(m_p);
			break;
		case Qt::Key_W:
			m_p[m_activeV * 3 + 1] += 0.4f;
			m_smr->updateRegion(m_p);
			break;
		case Qt::Key_S:
			m_p[m_activeV * 3 + 1] -= 0.4f;
			m_smr->updateRegion(m_p);
			break;
		case Qt::Key_E:
			m_p[m_activeV * 3 + 2] -= 0.4f;
			m_smr->updateRegion(m_p);
			break;
		case Qt::Key_R:
			m_p[m_activeV * 3 + 2] += 0.4f;
			m_smr->updateRegion(m_p);
			break;
		case Qt::Key_N:
			selectV(-1);
			break;
		case Qt::Key_M:
			selectV(1);
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
					2.f, 20.f, 34.64101616f, 1.f};
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

void GLWidget::selectV(int x)
{
	m_activeV += x;
	if(m_activeV > 5) 
		m_activeV = 0;
	if(m_activeV < 0) 
		m_activeV = 5;
}
