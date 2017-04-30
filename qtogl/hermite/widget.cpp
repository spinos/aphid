#include <QtGui>
#include <QtOpenGL>
#include "widget.h"
#include <math/linspace.h>
#include <GeoDrawer.h>
#include <ogl/DrawArrow.h>
#include <math/HermiteInterpolatePiecewise.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{
	m_interp = new HermiteInterpolatePiecewise<float, aphid::Vector3F >(3);
	
static const float P0[3][3] = {
{0,1,0},
{2,2,0},
{3,1,0}
};

static const float T0[3][3] = {
{2,-2,0},
{.5,0,0},
{1.5,-1.5,0}
};

static const float P1[3][3] = {
{2,2,0},
{3,1,0},
{3.3,0,0}
};

static const float T1[3][3] = {
{.5,0,0},
{1.5,-1.5,0},
{0,-1,0}
};

	for(int i=0;i<3;++i) {
		m_interp->setPieceBegin(i, P0[i], T0[i]);
		m_interp->setPieceEnd(i, P1[i], T1[i]);
	}
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{
}

void GLWidget::clientDraw()
{
	//getDrawer()->m_wireProfile.apply();
	getDrawer()->m_surfaceProfile.apply();

	getDrawer()->setColor(0.f, .4f, 1.f);
	
	float X[50];
	linspace<float>(X, 0.f, 1.f, 50);
	Vector3F Y[50];
	
	glBegin(GL_POINTS);
	for(int j=0;j<3;++j) {
		for(int i=0;i<50;++i) {
			Vector3F p = m_interp->interpolate(j, X[i]);
			glVertex3fv((float *)&p);
		}
	}
	glEnd();

}
