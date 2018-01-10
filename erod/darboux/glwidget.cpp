/*
 *  darboux
 */
#include <QtGui>
#include <QtOpenGL>
#include <math/BaseCamera.h>
#include "glwidget.h"
#include "TestContext.h"
#include <GeoDrawer.h>
#include <math/Quaternion.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	perspCamera()->setFarClipPlane(2000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(2000.f);
	orthoCamera()->setNearClipPlane(1.f);
	usePerspCamera();
	resetView();
	m_ctx = new TestContext;
}

GLWidget::~GLWidget()
{
}

void GLWidget::clientInit()
{
}

void GLWidget::clientDraw()
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
	const pbd::ParticleData* particle = m_ctx->c_particles();
	const Vector3F * pos = particle->pos();
	const int& np = particle->numParticles();
	glColor3f(1,1,1);
	glBegin(GL_LINES);
	for(int i=0; i< np-1;++i) {
		const Vector3F& p1 = pos[i];
		const Vector3F& p2 = pos[i+1];
		glVertex3f(p1.x,p1.y,p1.z);
		glVertex3f(p2.x,p2.y,p2.z);
	}
	glEnd();
	
	const pbd::ParticleData* ghost = m_ctx->c_ghostParticles();
	const Vector3F * gpos = ghost->pos();
	const int& ngp = ghost->numParticles();
	glColor3f(1,1,0);
	glBegin(GL_POINTS);
	for(int i=0; i< ngp;++i) {
		const Vector3F& p1 = gpos[i];
		glVertex3f(p1.x,p1.y,p1.z);
	}
	glEnd();
	
	Matrix44F dA, dB;
	Vector3F darboux;
	Vector3F corrv[5];
	m_ctx->getMaterialFrames(dA, dB, darboux, corrv);
	
	getDrawer()->coordsys1(dA);
	getDrawer()->coordsys1(dB);
	getDrawer()->setColor(1,1,0);
	getDrawer()->arrow(pos[1], pos[1] + darboux);
	
	glBegin(GL_LINES);
	for(int i=0; i< 3;++i) {
		const Vector3F& p1 = pos[i];
		const Vector3F& p2 = p1 + corrv[i];
		glVertex3f(p1.x,p1.y,p1.z);
		glVertex3f(p2.x,p2.y,p2.z);
	}
	for(int i=0; i< 2;++i) {
		const Vector3F& p1 = gpos[i];
		const Vector3F& p2 = p1 + corrv[i+3];
		glVertex3f(p1.x,p1.y,p1.z);
		glVertex3f(p2.x,p2.y,p2.z);
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
		case Qt::Key_K:
			rotateFrame(-0.07, Vector3F::ZAxis);
			break;
		case Qt::Key_L:
			rotateFrame(0.07, Vector3F::ZAxis);
			break;
		case Qt::Key_N:
			rotateFrame(-0.07, Vector3F::YAxis);
			break;
		case Qt::Key_M:
			rotateFrame(0.07, Vector3F::YAxis);
			break;
		case Qt::Key_V:
			rotateFrame(-0.07, Vector3F::XAxis);
			break;
		case Qt::Key_B:
			rotateFrame(0.07, Vector3F::XAxis);
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

void GLWidget::rotateFrame(float ang, const Vector3F& axis)
{
	Quaternion qrot(ang, axis);
	m_ctx->rotateFrame(qrot);
	update();
}
