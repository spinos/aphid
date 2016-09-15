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

using namespace aphid;

Ft2Widget::Ft2Widget(QWidget *parent) : Plot2DWidget(parent)
{
	UniformPlot2DImage * Xp = new UniformPlot2DImage;
#define DIM_Y 128
#define DIM_X 64
#define DIM_Z 1
	Xp->create(DIM_Y, DIM_X, DIM_Z);
	
	int i, j, k;
	for(k=0;k<DIM_Z;++k) {
		float * Xv = Xp->y(k);
		
	for(j=0;j<DIM_X;++j) {
		for(i=0;i<DIM_Y;++i) {
			Xv[j*DIM_Y+i] = RandomF01();
		}
	}
	}
	
	Xp->setDrawScale(2.f);
	Xp->updateImage();
	
	addImage(Xp);
	
	UniformPlot2DImage * lowpassP = new UniformPlot2DImage;
	lowpassP->create(DIM_Y/2, DIM_X, DIM_Z);
	
	UniformPlot2DImage * highpassP = new UniformPlot2DImage;
	highpassP->create(DIM_Y/2, DIM_X, DIM_Z);
	
	for(k=0;k<DIM_Z;++k) {
		
		wla::afbRow(Xp->channel(k), 
			lowpassP->channel(k), highpassP->channel(k) );
		
	}
	
	lowpassP->setDrawScale(2.f);
	lowpassP->updateImage();
	
	addImage(lowpassP);
	
	highpassP->setDrawScale(2.f);
	highpassP->updateImage(true);
	
	addImage(highpassP);
}

Ft2Widget::~Ft2Widget()
{}

