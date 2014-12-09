#include <QtGui>
#include <QtOpenGL>

#include "glwidget.h"
#include <cmath>

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	m_vertexBuffer = new CUDABuffer;
	m_program = new BezierProgram;
}

GLWidget::~GLWidget()
{
}

void GLWidget::clientInit()
{
    CUDABuffer::setDevice();
}

void GLWidget::clientDraw()
{
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
