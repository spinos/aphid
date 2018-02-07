/*
 *  sdft
 */
#include <GeoDrawer.h>
#include <QtGui>
#include <QtOpenGL>
#include "widget.h"
#include <ogl/DrawCircle.h>
#include <ogl/RotationHandle.h>
#include <ogl/TranslationHandle.h>
#include <ogl/ScalingHandle.h>
#include <math/BaseCamera.h>
#include <ogl/DrawBox.h>
#include <ogl/DrawDop.h>
#include <ogl/DrawArrow.h>
#include "LegendreDFTest.h"
#include <geom/SuperShape.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{ 
	usePerspCamera(); 	
	m_legen = new LegendreDFTest;
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{
	m_space.translate(1,1,1);
	m_legen->init();
}

void GLWidget::clientDraw()
{
	getDrawer()->m_surfaceProfile.apply();
	
	getDrawer()->m_markerProfile.apply();
	
	m_legen->draw(getDrawer() );
	
	getDrawer()->m_wireProfile.apply();
	getDrawer()->setColor(.1f, .3f, .4f);
	m_legen->drawShape(getDrawer() );
}

void GLWidget::clientSelect(QMouseEvent *event)
{
	m_incident = *getIncidentRay();
	m_legen->rayIntersect(getIncidentRay() );
	update();
}

void GLWidget::clientDeselect(QMouseEvent *event)
{
	update();
}

void GLWidget::clientMouseInput(QMouseEvent *event)
{
	m_incident = *getIncidentRay();
	m_legen->rayIntersect(getIncidentRay() );
	update();
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	switch (e->key()) {
		case Qt::Key_M:
			m_legen->measureShape();
			update();
			break;
		case Qt::Key_L:
			break;
		case Qt::Key_P:
			break;
		default:
			break;
	}
	Base3DView::keyPressEvent(e);
}

void GLWidget::receiveA1(double x)
{ 
	m_legen->shapeParam()._a1 = x;
	m_legen->updateShape();
	update();
}

void GLWidget::receiveB1(double x)
{ 
	m_legen->shapeParam()._b1 = x;
	m_legen->updateShape();
	update();
}

void GLWidget::receiveM1(double x)
{ 
	m_legen->shapeParam()._m1 = x;
	m_legen->updateShape();
	update();
}

void GLWidget::receiveN1(double x)
{ 
	m_legen->shapeParam()._n1 = x;
	m_legen->updateShape();
	update();
}

void GLWidget::receiveN2(double x)
{ 
	m_legen->shapeParam()._n2 = x;
	m_legen->updateShape();
	update();
}

void GLWidget::receiveN3(double x)
{ 
	m_legen->shapeParam()._n3 = x;
	m_legen->updateShape();
	update();
}

void GLWidget::receiveA2(double x)
{ 
	m_legen->shapeParam()._a2 = x;
	m_legen->updateShape();
	update();
}

void GLWidget::receiveB2(double x)
{ 
	m_legen->shapeParam()._b2 = x;
	m_legen->updateShape();
	update();
}

void GLWidget::receiveM2(double x)
{ 
	m_legen->shapeParam()._m2 = x;
	m_legen->updateShape();
	update();
}

void GLWidget::receiveN21(double x)
{ 
	m_legen->shapeParam()._n21 = x;
	m_legen->updateShape();
	update();
}

void GLWidget::receiveN22(double x)
{ 
	m_legen->shapeParam()._n22 = x;
	m_legen->updateShape();
	update();
}

void GLWidget::receiveN23(double x)
{ 
	m_legen->shapeParam()._n23 = x;
	m_legen->updateShape();
	update();
}
