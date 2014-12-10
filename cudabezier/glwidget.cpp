#include <QtGui>
#include <QtOpenGL>

#include "glwidget.h"
#include <KdTreeDrawer.h>

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	m_vertexBuffer = new CUDABuffer;
	m_cvs = new CUDABuffer;
	m_program = new BezierProgram;
	m_curve = new BaseCurve;
	m_curve->createVertices(4);
	m_curve->m_cvs[0] = Vector3F(-10.f, 0.f, 0.f);
	m_curve->m_cvs[1] = Vector3F(-10.f, 20.f, 0.f);
	m_curve->m_cvs[2] = Vector3F(10.f, 20.f, 0.f);
	m_curve->m_cvs[3] = Vector3F(10.f, 0.f, 0.f);
}

GLWidget::~GLWidget()
{
}

void GLWidget::clientInit()
{
    CUDABuffer::setDevice();
	m_vertexBuffer->create(0, m_curve->numSegments() * 100 * 16);
	m_cvs->create(m_curve->numVertices() * 12);
}

void GLWidget::clientDraw()
{
	// m_vertexBuffer->create(m_data, m_curve->numSegments() * 100 * 16);
	m_program->run(m_vertexBuffer, m_cvs, m_curve);
	// GeoDrawer * dr = getDrawer();
	// dr->linearCurve(*m_curve);
	// qDebug()<<" "<<m_vertexBuffer->bufferSize()<<" "<<m_vertexBuffer->_buffereName;
	glBindBuffer(GL_ARRAY_BUFFER, (GLuint)m_vertexBuffer->bufferName());
    glVertexPointer(4, GL_FLOAT, 0, 0);
	
	glEnableClientState(GL_VERTEX_ARRAY);
	
	glDrawArrays(GL_POINTS, 0, m_curve->numSegments() * 100);

	glDisableClientState(GL_VERTEX_ARRAY);
	
	// glBindBuffer(GL_ARRAY_BUFFER, 0);
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
