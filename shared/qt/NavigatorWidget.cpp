/*
 *   NavigatorWidget.cpp
 *
 */
 
#include <QtGui>
#include "NavigatorWidget.h"
#include <img/ExrImage.h>

namespace aphid {

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

void NavigatorWidget::processCamera(QMouseEvent *event)
{}

}
