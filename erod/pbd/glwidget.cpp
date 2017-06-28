#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "glwidget.h"
#include <SolverThread.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
    qDebug()<<"glview";
	perspCamera()->setFarClipPlane(2000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(2000.f);
	orthoCamera()->setNearClipPlane(1.f);
	usePerspCamera();
	resetView();
	m_solver = new SolverThread;
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
}

void GLWidget::clientInit()
{
    m_solver->initProgram();

    connect(internalTimer(), SIGNAL(timeout()), m_solver, SLOT(simulate()));
	connect(m_solver, SIGNAL(doneStep()), this, SLOT(update()));
}

void GLWidget::clientDraw()
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
	const Vector3F * pos = m_solver->pos();
	const unsigned * indices = m_solver->indices();
	const unsigned NI = m_solver->numIndices();
	glColor3f(1,1,1);
	glBegin(GL_TRIANGLES);
	unsigned i;
	for(i=0; i< NI; i += 3) {
		Vector3F p1 = pos[indices[i]];
		Vector3F p2 = pos[indices[i+1]];
		Vector3F p3 = pos[indices[i+2]];
		glVertex3f(p1.x,p1.y,p1.z);
		glVertex3f(p2.x,p2.y,p2.z);
		glVertex3f(p3.x,p3.y,p3.z);
	}
	glEnd();
	
	/*
	Vector3F bbox[NTri * 2];
    m_program->getAabbs(bbox, NTri);

	GeoDrawer * dr = getDrawer();
	dr->setColor(0.f, 0.5f, 0.f);
	for(i=0; i< NTri; i++) {
	    BoundingBox bb;
	    bb.updateMin(bbox[i*2]);
	    bb.updateMax(bbox[i*2 + 1]);
	    dr->boundingBox(bb);
	}
	
	glColor3f(1,0,0);
	glBegin(GL_LINES);
	for(i=0; i< m_numSpring; i++) {
		// if(m_spring[i].type != BEND_SPRING) continue;
		Vector3F p1 = m_pos[m_spring[i].p1];
		Vector3F p2 = m_pos[m_spring[i].p2];
		glVertex3f(p1.x,p1.y,p1.z);
		glVertex3f(p2.x,p2.y,p2.z);
	}
	glEnd();*/
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
