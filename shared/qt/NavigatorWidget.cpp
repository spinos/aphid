/*
 *   NavigatorWidget.cpp
 *
 */
 
#include <QtGui>
#include "NavigatorWidget.h"
#include <img/ExrImage.h>

namespace aphid {

NavigatorWidget::NavigatorWidget(QWidget *parent) : Plot2DWidget(parent)
{
	setMargin(4, 4);
	
	setMinimumWidth(264);
	setMinimumHeight(264);
	
	QSizePolicy spc(QSizePolicy::Preferred, QSizePolicy::Fixed);
	setSizePolicy(spc);
	
	Array3<float> sigx;
	sigx.create(256, 256, 3);
	sigx.rank(0)->set(0.f);
	sigx.rank(1)->set(0.f);
	sigx.rank(2)->set(0.f);
	
	UniformPlot2DImage * yp1 = new UniformPlot2DImage;
	yp1->create(sigx);
	yp1->setDrawScale(1.f);				
	yp1->updateImage();
	
	addImage(yp1);
}

NavigatorWidget::NavigatorWidget(const ExrImage * img,
					QWidget *parent) : Plot2DWidget(parent)
{
	setMargin(4, 4);
	
	setMinimumWidth(264);
	setMinimumHeight(264);
	
	QSizePolicy spc(QSizePolicy::Preferred, QSizePolicy::Fixed);
	setSizePolicy(spc);
	
	if(!img->isValid() ) {
		return;
	}
	
	int xdim, ydim;
	img->getThumbnailSize(xdim, ydim);
	Array3<float> sigx;
	sigx.create(xdim, ydim, 1);
	img->resampleRed(sigx.rank(0)->v(), xdim, ydim);
	
	UniformPlot2DImage * yp1 = new UniformPlot2DImage;
	yp1->create(sigx);
	yp1->setDrawScale(1.f);				
	yp1->updateImage();
	
	addImage(yp1);
}

NavigatorWidget::~NavigatorWidget()
{}

void NavigatorWidget::setImage(const Array3<float> & img)
{
	UniformPlot2DImage * yp1 = plotImage(0);
	
	yp1->create(img);
	yp1->updateImage();
	
}

void NavigatorWidget::processCamera(QMouseEvent *event)
{}

}
