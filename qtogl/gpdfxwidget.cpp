/*
 *  gpdfxwidget.cpp
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "gpdfxwidget.h"
#include "cactus.h"

using namespace aphid;

GpdfxWidget::GpdfxWidget(QWidget *parent) : Plot1DWidget(parent)
{
	setBound(-1.5, 1.5, 6, -1.5, 1.5, 6);
	int dim = sCactusNumObservations;
	UniformPlot1D * ap = new UniformPlot1D;
	ap->create(dim);
	
	int i=0, j;
	for(;i<dim;++i) {
		ap->x()[i] = sCactusXValues[i][0];
		ap->y()[i] = sCactusXValues[i][1];
	}
	
	ap->setGeomType(UniformPlot1D::GtMark);
	ap->setColor(0,.5,1);
	addVectorPlot(ap);
	
	if(parent) {
		connect(this, SIGNAL(xValueChanged(QPointF) ), 
			parent, SLOT(recvXValue(QPointF) ) );
	}
	
	bp = new UniformPlot1D;
	bp->create(1);
	bp->x()[0] = 0.0;
	bp->y()[0] = 0.0;
	bp->setGeomType(UniformPlot1D::GtMark);
	bp->setColor(.85,.5,0.);
	addVectorPlot(bp);
}

GpdfxWidget::~GpdfxWidget()
{}

void GpdfxWidget::mousePressEvent(QMouseEvent *event)
{
	Vector2F vmouse = toRealSpace(event->x(), event->y());
	emit xValueChanged(QPointF(vmouse.x, vmouse.y));
	bp->x()[0] = vmouse.x;
	bp->y()[0] = vmouse.y;
	update();
}

void GpdfxWidget::mouseMoveEvent(QMouseEvent *event)
{
	Vector2F vmouse = toRealSpace(event->x(), event->y());
	emit xValueChanged(QPointF(vmouse.x, vmouse.y));
	bp->x()[0] = vmouse.x;
	bp->y()[0] = vmouse.y;
	update();
}
