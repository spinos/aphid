/*
 *  rodt
 */
#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "glwidget.h"
#include <SolverThread.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	perspCamera()->setFarClipPlane(2000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(2000.f);
	orthoCamera()->setNearClipPlane(1.f);
	usePerspCamera();
	resetView();
	m_solver = new SolverThread;
}

GLWidget::~GLWidget()
{
}

void GLWidget::clientInit()
{
    connect(internalTimer(), SIGNAL(timeout()), m_solver, SLOT(simulate()));
	connect(m_solver, SIGNAL(doneStep()), this, SLOT(update()));
}

void GLWidget::clientDraw()
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
	const pbd::ParticleData* particle = m_solver->c_particles();
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
	
	const pbd::ParticleData* ghost = m_solver->c_ghostParticles();
	const Vector3F * gpos = ghost->pos();
	const int& ngp = ghost->numParticles();
	glColor3f(1,1,0);
	glBegin(GL_POINTS);
	for(int i=0; i< ngp;++i) {
		const Vector3F& p1 = gpos[i];
		glVertex3f(p1.x,p1.y,p1.z);
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
	Base3DView::keyReleaseEvent(event);
}

void GLWidget::resetPerspViewTransform()
{
static const float mm[16] = {1.f, 0.f, 0.f, 0.f,
					0.f, 0.8660254f, -0.5f, 0.f,
					0.f, 0.5f, 0.8660254f, 0.f,
					32.f, 200.f, 346.4101616f, 1.f};
	Matrix44F mat(mm);
	perspCamera()->setViewTransform(mat, 400.f);
}

void GLWidget::resetOrthoViewTransform()
{
static const float mm1[16] = {1.f, 0.f, 0.f, 0.f,
					0.f, 0.8660254f, -0.5f, 0.f,
					0.f, 0.5f, 0.8660254f, 0.f,
					32.f, 200.f, 346.4101616f, 1.f};
	Matrix44F mat(mm1);
	orthoCamera()->setViewTransform(mat, 150.f);
	orthoCamera()->setHorizontalAperture(150.f);
}
