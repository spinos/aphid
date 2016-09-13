/*
 *  ft1dWidget.cpp
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "ft1dWidget.h"

using namespace aphid;

Ft1dWidget::Ft1dWidget(QWidget *parent) : Plot2DWidget(parent)
{
	UniformPlot1D * ap = new UniformPlot1D;
	ap->create(64);
	int i=0;
	for(;i<64;++i)
		ap->y()[i] = RandomF01();
		
	ap->setColor(0,0,1);
	addPlot(ap);

	float * Xshf = calc::circshift(ap->y(), 64, -5);
	float * Xflt = calc::firlHann(Xshf, 64);
	
	UniformPlot1D * fp = new UniformPlot1D;
	fp->create(Xflt, 64);
	fp->setColor(0,.5,.5);
	addPlot(fp);
	
	delete[] Xshf;
	delete[] Xflt;
	
	int nhwl;
	float * hwl = calc::hannLowPass(nhwl);
	
	UniformPlot1D * fhwl = new UniformPlot1D;
	fhwl->create(hwl, nhwl);
	fhwl->setColor(1.,0., 0.);
	addPlot(fhwl);
	
	delete[] hwl;
}

Ft1dWidget::~Ft1dWidget()
{}

