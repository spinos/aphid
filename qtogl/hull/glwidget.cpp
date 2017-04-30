#include <QtGui>
#include <QtOpenGL>
#include "glwidget.h"
#include "HullContainer.h"

using namespace aphid;

GLWidget::GLWidget(QWidget *parent)
    : Base3DView(parent)
{
	_dynamics = new HullContainer();
	_dynamics->initHull();
	
	std::cout<<" convex hull n horizon "<<_dynamics->getNumFace();
	std::cout.flush();
}

GLWidget::~GLWidget()
{}

void GLWidget::clientInit()
{}

void GLWidget::clientDraw()
{	
	//drawer.drawWiredFace(_dynamics);
	//drawer.setColor(0.f, .75f, 1.f);
	//drawer.drawNormal(_dynamics);
	//_dynamics->renderWorld(&drawer);
	
}
