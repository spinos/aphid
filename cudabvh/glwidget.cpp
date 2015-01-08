#include <QtGui>

#include <gl_heads.h>
#include "glwidget.h"

#include <KdTreeDrawer.h>
#include <CUDABuffer.h>
#include <BvhSolver.h>

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	m_solver = new BvhSolver;
}

GLWidget::~GLWidget()
{
}

void GLWidget::clientInit()
{
	m_solver->init();
	//m_cvs->create(m_curve->numVertices() * 12);
	//m_cvs->hostToDevice(m_curve->m_cvs, m_curve->numVertices() * 12);
	//m_program->run(m_vertexBuffer, m_cvs, m_curve);
	connect(internalTimer(), SIGNAL(timeout()), m_solver, SLOT(simulate()));
	connect(m_solver, SIGNAL(doneStep()), this, SLOT(update()));
	// connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
}

void GLWidget::clientDraw()
{
	// GeoDrawer * dr = getDrawer();
	// dr->linearCurve(*m_curve);
	m_solver->formPlane((float)elapsedTime()/320.f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glEnableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, (GLuint)m_solver->vertexBufferName());
    glVertexPointer(4, GL_FLOAT, 0, 0);
	
	//
	
	// glDrawArrays(GL_POINTS, 0, m_solver->numVertices());
	// glVertexPointer(4, GL_FLOAT, 0, (GLfloat*)m_solver->numVertices());
	
	glDrawElements(GL_TRIANGLES, m_solver->getNumTriangleFaceVertices(), GL_UNSIGNED_INT, m_solver->getIndices());

	glDisableClientState(GL_VERTEX_ARRAY);
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	// qDebug()<<"drawn in "<<deltaTime();
}

void GLWidget::clientSelect(QMouseEvent */*event*/)
{
	setUpdatesEnabled(false);
	
	setUpdatesEnabled(true);
}

void GLWidget::clientDeselect(QMouseEvent */*event*/) 
{
	setUpdatesEnabled(false);
	
	setUpdatesEnabled(true);
}

void GLWidget::clientMouseInput(QMouseEvent */*event*/)
{
	setUpdatesEnabled(false);
	
	setUpdatesEnabled(true);
}

void GLWidget::focusInEvent(QFocusEvent * event)
{
	qDebug()<<"focus in";
	Base3DView::focusInEvent(event);
}

void GLWidget::focusOutEvent(QFocusEvent * event)
{
	qDebug()<<"focus out";
	Base3DView::focusOutEvent(event);
}
