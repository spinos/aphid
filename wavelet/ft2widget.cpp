/*
 *  Ft2Widget.cpp
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "ft2widget.h"
#include "dwt2.h"
#include "gensig.h"

using namespace aphid;

Ft2Widget::Ft2Widget(QWidget *parent) : Plot2DWidget(parent)
{
	UniformPlot2DImage * Xp = new UniformPlot2DImage;
#define DIM_Y 256
#define DIM_X 256
#define DIM_Z 1
	gen2dsig(Xp, DIM_Y, DIM_X, DIM_Z);
	
	Xp->setDrawScale(1.f);
	Xp->updateImage();
	
	addImage(Xp);
	
	UniformPlot2DImage * lowP = new UniformPlot2DImage;
	lowP->create(DIM_Y/2, DIM_X/2, DIM_Z);
	
	UniformPlot2DImage * lowhighP = new UniformPlot2DImage;
	lowhighP->create(DIM_Y/2, DIM_X/2, DIM_Z);
	
	UniformPlot2DImage * highlowP = new UniformPlot2DImage;
	highlowP->create(DIM_Y/2, DIM_X/2, DIM_Z);
	
	UniformPlot2DImage * highhighP = new UniformPlot2DImage;
	highhighP->create(DIM_Y/2, DIM_X/2, DIM_Z);
	
	int k;
	for(k=0;k<DIM_Z;++k) {
		
		wla::afb2(Xp->channel(k), 
			lowP->channel(k), lowhighP->channel(k),
			highlowP->channel(k), highhighP->channel(k) );
		
	}
	
	lowP->setDrawScale(1.f);
	lowP->updateImage();
	
	
	lowhighP->setDrawScale(1.f);
	lowhighP->updateImage(true);
	
	highlowP->setDrawScale(1.f);
	highlowP->updateImage(true);
	
	highhighP->setDrawScale(1.f);
	highhighP->updateImage(true);
	
	UniformPlot2DImage * yP = new UniformPlot2DImage;
	yP->create(DIM_Y, DIM_X, DIM_Z);
	
	for(k=0;k<DIM_Z;++k) {
		
		wla::sfb2(yP->channel(k), 
			lowP->channel(k), lowhighP->channel(k),
			highlowP->channel(k), highhighP->channel(k) );
		
	}
	
	yP->setDrawScale(1.f);
	yP->updateImage();
	
	addImage(yP);
	addImage(lowP);
	addImage(lowhighP);
	addImage(highlowP);
	addImage(highhighP);
	
	float mxe = 0.f;
	
	for(k=0;k<DIM_Z;++k) {
/// original x
		wla::circleShift2(Xp->channel(k), 5);
		Xp->channel(k)->maxAbsError(mxe, *yP->channel(k) );
		
	}
	
	std::cout<<"\n max err "<<mxe;
	std::cout.flush();
}

Ft2Widget::~Ft2Widget()
{}

