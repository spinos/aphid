#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include <SolverThread.h>

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
    qDebug()<<"glview";
	perspCamera()->setFarClipPlane(2000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(2000.f);
	orthoCamera()->setNearClipPlane(1.f);
	
	m_solver = new SolverThread;
}
//! [0]

//! [1]
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
    unsigned nt = m_solver->numTetrahedrons();
	Tetrahedron * t = m_solver->tetrahedron();
	Vector3F * p = m_solver->X();
	Vector3F a, b, c, d;
	glBegin(GL_LINES);
	for(unsigned i = 0; i < nt; i++) {
	    Tetrahedron tet = t[i];
	    a = p[tet.indices[0]];
	    b = p[tet.indices[1]];
	    c = p[tet.indices[2]];
	    d = p[tet.indices[3]];
	    glVertex3f(a.x, a.y, a.z);
	    glVertex3f(b.x, b.y, b.z);
	    
	    glVertex3f(a.x, a.y, a.z);
	    glVertex3f(c.x, c.y, c.z);
	    
	    glVertex3f(a.x, a.y, a.z);
	    glVertex3f(d.x, d.y, d.z);
	    
	    glVertex3f(b.x, b.y, b.z);
	    glVertex3f(c.x, c.y, c.z);
	    
	    glVertex3f(c.x, c.y, c.z);
	    glVertex3f(d.x, d.y, d.z);
	    
	    glVertex3f(d.x, d.y, d.z);
	    glVertex3f(b.x, b.y, b.z);
	    
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
