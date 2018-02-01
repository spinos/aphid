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
		case Qt::Key_K:
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
