#include <QtGui>
#include <QtOpenGL>
#include "widget.h"
#include <GeoDrawer.h>
#include <ogl/DrawCircle.h>
#include <ogl/RotationHandle.h>
#include <BaseCamera.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{ 
	usePerspCamera(); 	
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{
	m_space.translate(1,1,1);
	m_roth = new RotationHandle(&m_space);
	
}

void GLWidget::clientDraw()
{
	getDrawer()->m_markerProfile.apply();
	//getDrawer()->m_surfaceProfile.apply();

	//getDrawer()->setColor(0.f, .34f, .45f);

	float m[16];
	m_space.glMatrix(m);
	DrawCircle dc;
	dc.draw3Circles(m);
	
	Vector3F veye = getCamera()->eyeDirection();
	Matrix44F meye = m_space;
	meye.setFrontOrientation(veye );
	meye.glMatrix(m);

	m_roth->draw(m);
	
}

void GLWidget::clientSelect(QMouseEvent *event)
{
	m_roth->begin(getIncidentRay() );
	update();
}

void GLWidget::clientDeselect(QMouseEvent *event)
{
	m_roth->end();
	update();
}

void GLWidget::clientMouseInput(QMouseEvent *event)
{
	m_roth->rotate(getIncidentRay() );
	
	update();
}
	