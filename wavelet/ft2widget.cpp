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
#include "fltbanks.h"

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
	
	UniformPlot2DImage * Xsfp = new UniformPlot2DImage;
	Xsfp->create(DIM_Y, DIM_X, DIM_Z);
	for(k=0;k<DIM_Z;++k) {
		
		*Xsfp->channel(k) = *Xp->channel(k);
		
	}
	
	Xsfp->updateImage();
	
	addImage(Xsfp);
}

Ft2Widget::~Ft2Widget()
{}

