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
#include <math/AOrientedBox.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{ 
	usePerspCamera(); 	
	m_hil.init();
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{
	m_space.translate(1,1,1);
}

void GLWidget::clientDraw()
{
	getDrawer()->m_surfaceProfile.apply();
	
	getDrawer()->m_markerProfile.apply();
	m_hil.draw(getDrawer() );

	glBegin(GL_LINES);
	glVertex3fv((const GLfloat*)&m_incident.travel(2.f) );
	glVertex3fv((const GLfloat*)&m_incident.travel(1000.f) );
	glEnd();
	
	getDrawer()->cube(m_incident.travel(2.f), .25f);
	
}

void GLWidget::clientSelect(QMouseEvent *event)
{
	m_incident = *getIncidentRay();
	update();
}

void GLWidget::clientDeselect(QMouseEvent *event)
{
	update();
}

void GLWidget::clientMouseInput(QMouseEvent *event)
{
	m_incident = *getIncidentRay();
	update();
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	switch (e->key()) {
		case Qt::Key_K:
			m_hil.progressBackward();
			break;
		case Qt::Key_L:
			m_hil.progressForward();
			break;
		case Qt::Key_P:
			m_hil.printCoord();
			break;
		default:
			break;
	}
	Base3DView::keyPressEvent(e);
}
