#include <QtGui>
#include <GeoDrawer.h>
#include <QtOpenGL>
#include "widget.h"
#include <geom/Airfoil.h>
#include <math/linspace.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{
	m_cpt.set(.02f, .4f, .15f);
}

void GLWidget::clientDraw()
{
    getDrawer()->m_wireProfile.apply();

	getDrawer()->setColor(0.f, .34f, .55f);

	Airfoil airfoil(32.f, m_cpt.x, m_cpt.y, m_cpt.z);
	const float & ch = airfoil.chord();
	
	const int n = 50;
	float * yc = new float[n];
	float * yt = new float[n];
	float * x = new float[n];
	linspace<float>(x, 0.f, ch, n);
	for(int i=0;i<n;++i) {
		yc[i] = airfoil.calcYc(x[i] / ch);
		yt[i] = airfoil.calcYt(x[i] / ch);
	}	
	
	glBegin(GL_LINE_STRIP);
	
	for(int i=0;i<n;++i) {
		glVertex3f(x[i], yc[i], 0.f);
	}
	
	glEnd();
	
	getDrawer()->setColor(0.f, 0.f, .55f);
	
	glBegin(GL_LINE_STRIP);
	
	for(int i=0;i<n;++i) {
		float theta = airfoil.calcTheta(x[i]);
		glVertex3f(x[i] - yt[i] * sin(theta), yc[i] + yt[i] * cos(theta), 0.f);
	}
	
	glEnd();
	
	glBegin(GL_LINE_STRIP);
	
	for(int i=0;i<n;++i) {
		float theta = airfoil.calcTheta(x[i]);
		glVertex3f(x[i] + yt[i] * sin(theta), yc[i] - yt[i]* cos(theta), 0.f);
	}
	
	glEnd();
	
	delete[] x;
	delete[] yc;
	delete[] yt;
}

void GLWidget::recvParam(Float3 vx)
{ 
	m_cpt = vx;
	update();
}
