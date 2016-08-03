#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "glwidget.h"
#include <GeoDrawer.h>
#include "SuperformulaTest.h"
#include "RedBlueTest.h"

using namespace aphid;

namespace ttg {

GLWidget::GLWidget(ttg::Scene * sc, QWidget *parent) : Base3DView(parent)
{
	m_scene = sc;
	m_scene->setView(perspectiveView() );
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
}

void GLWidget::clientInit()
{
	m_scene->init();
	if(m_scene->viewPerspective() )
		usePerspCamera();
	//connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update()));
}

void GLWidget::clientDraw()
{
	updatePerspectiveView();
	getDrawer()->frustum(perspectiveView()->frustum() );
	m_scene->draw(getDrawer() );
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

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	switch (e->key()) {
		case Qt::Key_M:
			m_scene->progressForward();
			break;
		case Qt::Key_N:
			m_scene->progressBackward();
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

void GLWidget::receiveA1(double x)
{
	reinterpret_cast<SuperformulaTest *> (m_scene)->setA1(x);
	update();
}

void GLWidget::receiveB1(double x)
{
	reinterpret_cast<SuperformulaTest *> (m_scene)->setB1(x);
	update();
}

void GLWidget::receiveM1(double x)
{
	reinterpret_cast<SuperformulaTest *> (m_scene)->setM1(x);
	update();
}

void GLWidget::receiveN1(double x)
{
	reinterpret_cast<SuperformulaTest *> (m_scene)->setN1(x);
	update();
}

void GLWidget::receiveN2(double x)
{
	reinterpret_cast<SuperformulaTest *> (m_scene)->setN2(x);
	update();
}

void GLWidget::receiveN3(double x)
{
	reinterpret_cast<SuperformulaTest *> (m_scene)->setN3(x);
	update();
}

void GLWidget::receiveA2(double x)
{
	reinterpret_cast<SuperformulaTest *> (m_scene)->setA2(x);
	update();
}

void GLWidget::receiveB2(double x)
{
	reinterpret_cast<SuperformulaTest *> (m_scene)->setB2(x);
	update();
}

void GLWidget::receiveM2(double x)
{
	reinterpret_cast<SuperformulaTest *> (m_scene)->setM2(x);
	update();
}

void GLWidget::receiveN21(double x)
{
	reinterpret_cast<SuperformulaTest *> (m_scene)->setN21(x);
	update();
}

void GLWidget::receiveN22(double x)
{
	reinterpret_cast<SuperformulaTest *> (m_scene)->setN22(x);
	update();
}

void GLWidget::receiveN23(double x)
{
	reinterpret_cast<SuperformulaTest *> (m_scene)->setN23(x);
	update();
}

void GLWidget::receiveA(double x)
{
	reinterpret_cast<RedBlueTest *> (m_scene)->setA(x);
	update();
}

void GLWidget::receiveB(double x)
{
	reinterpret_cast<RedBlueTest *> (m_scene)->setB(x);
	update();
}

void GLWidget::receiveC(double x)
{
	reinterpret_cast<RedBlueTest *> (m_scene)->setC(x);
	update();
}

void GLWidget::receiveD(double x)
{
	reinterpret_cast<RedBlueTest *> (m_scene)->setD(x);
	update();
}
	
}