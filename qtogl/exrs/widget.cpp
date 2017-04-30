#include <QtGui>
#include <QtOpenGL>
#include "widget.h"
#include <GeoDrawer.h>
#include <ogl/DrawCircle.h>
#include <ogl/RotationHandle.h>
#include <ogl/TranslationHandle.h>
#include <ogl/ScalingHandle.h>
#include <BaseCamera.h>
#include <ogl/DrawBox.h>
#include <ogl/DrawDop.h>
#include <ogl/DrawArrow.h>
#include <math/miscfuncs.h>
#include <img/ExrImage.h>

using namespace aphid;

GLWidget::GLWidget(const std::string & fileName, QWidget *parent)
    : Base3DView(parent)
{ 
	usePerspCamera(); 
	m_sampler = new ExrImage;
	m_sampler->read(fileName);
	
	if(m_sampler->isValid() ) {
		sampleImage();
	} else {
		std::cout<<"\n ERROR image not valid "<<fileName;
		std::cout.flush();
	}
}

GLWidget::~GLWidget()
{}

void GLWidget::sampleImage()
{
	float s, t;
	for(int i=0;i<NUM_SMP;++i) {
		s = RandomF01();
		t = RandomF01();
		m_pos[i].set(s * 20.f - 10.f, t * 20.f - 10.f, 0.f);
		float * ci = (float *)&m_col[i];
		m_sampler->sample(s, t, 3, ci);
		m_col[i].w = 1.f;
	}
}

void GLWidget::clientInit()
{}

void GLWidget::clientDraw()
{
	getDrawer()->m_surfaceProfile.apply();
	
	getDrawer()->m_markerProfile.apply();
	
	glBegin(GL_POINTS);
	for(int i=0;i<NUM_SMP;++i) {
		glColor3fv((const float *)&m_col[i]);
		glVertex3fv((const float *)&m_pos[i]);
	}
	glEnd();
}

void GLWidget::clientSelect(QMouseEvent *event)
{}

void GLWidget::clientDeselect(QMouseEvent *event)
{}

void GLWidget::clientMouseInput(QMouseEvent *event)
{}
