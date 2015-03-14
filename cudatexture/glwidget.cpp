#include <QtGui>

#include <gl_heads.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include <CudaBase.h>
#include <CudaTexture.h>
#include <CUDABuffer.h>
#include "simple_implement.h"

#define TEX_SIZE 1024

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
    m_tex = new CudaTexture;
    m_buf = new CUDABuffer;
}

GLWidget::~GLWidget()
{
}

void GLWidget::clientInit()
{
	CudaBase::SetDevice();
	m_tex->create(TEX_SIZE, TEX_SIZE, 4, false);
	m_buf->create(TEX_SIZE * TEX_SIZE * 16);
	connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
}

void GLWidget::clientDraw()
{
    float4 * p = (float4 *)m_buf->bufferOnDevice();
    fillImage(p, TEX_SIZE * TEX_SIZE);
    
    m_tex->copyFrom(m_buf->bufferOnDevice(), TEX_SIZE * TEX_SIZE * 16);
    
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, *m_tex->texture());
    
    glColor3f(1.f, 1.f, 1.f);
    glBegin(GL_TRIANGLES);
    glTexCoord2f(0.f, 0.f);
    glVertex3f(0.f, 0.f, 0.f);
    glTexCoord2f(1.f, 0.f);
    glVertex3f(10.f, 0.f, 0.f);
    glTexCoord2f(1.f, 1.f);
    glVertex3f(10.f, 10.f, 0.f);
    
    glTexCoord2f(0.f, 0.f);
    glVertex3f(0.f, 0.f, 0.f);
    glTexCoord2f(1.f, 1.f);
    glVertex3f(10.f, 10.f, 0.f);
    glTexCoord2f(0.f, 1.f);
    glVertex3f(0.f, 10.f, 0.f);
    
    glEnd();
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

void GLWidget::keyPressEvent(QKeyEvent *event)
{
	switch (event->key()) {
		case Qt::Key_A:
			break;
		case Qt::Key_D:
			break;
		case Qt::Key_W:
			internalTimer()->stop();
			break;
		case Qt::Key_S:
			internalTimer()->start();
			break;
		default:
			break;
	}
	
	Base3DView::keyPressEvent(event);
}
