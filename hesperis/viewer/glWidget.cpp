/*
 *  widget.cpp
 *  hes viewer
 *
 */
#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include <PerspectiveView.h>
#include "glWidget.h"

using namespace aphid;

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent),
m_dspState(0)
{
	perspCamera()->setFarClipPlane(30000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(30000.f);
	orthoCamera()->setNearClipPlane(1.f);
	usePerspCamera();
	resetView();
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{}

void GLWidget::clientDraw()
{
}

void GLWidget::resetPerspViewTransform()
{
static const float mm[16] = {1.f, 0.f, 0.f, 0.f,
					0.f, 0.8660254f, -0.5f, 0.f,
					0.f, 0.5f, 0.8660254f, 0.f,
					0.f, 200.f, 346.4101616f, 1.f};
	Matrix44F mat(mm);
	perspCamera()->setViewTransform(mat, 400.f);
}

void GLWidget::resetOrthoViewTransform()
{
static const float mm1[16] = {1.f, 0.f, 0.f, 0.f,
					0.f, 0.f, -1.f, 0.f,
					0.f, 1.f, 0.f, 0.f,
					0.f, 150.f, 0.f, 1.f};
	Matrix44F mat(mm1);
	orthoCamera()->setViewTransform(mat, 150.f);
	orthoCamera()->setHorizontalAperture(150.f);
}

void GLWidget::clientSelect(Vector3F & origin, Vector3F & ray, Vector3F & hit)
{
}

void GLWidget::clientDeselect()
{
}

void GLWidget::clientMouseInput(Vector3F & stir)
{
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	switch (e->key()) {
		case Qt::Key_W:
			break;
		case Qt::Key_N:
			break;
		default:
			break;
	}
	Base3DView::keyPressEvent(e);
}

void GLWidget::keyReleaseEvent(QKeyEvent *event)
{
	Base3DView::keyReleaseEvent(event);
}
	
void GLWidget::recvToolAction(int x)
{
	qDebug()<<" unknown tool action "<<x;
}

void GLWidget::setDisplayState(int x)
{
	m_dspState = x;
	update();
}