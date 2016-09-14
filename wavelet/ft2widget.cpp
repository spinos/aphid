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
	Xp->create(128, DIM_X);
	float * Xv = Xp->y();
	
	int i, j;
	for(j=0;j<DIM_X;++j) {
		for(i=0;i<DIM_Y;++i) {
			Xv[j*DIM_Y+i] = RandomF01();
		}
	}
	
	Xp->setDrawScale(2.f);
	Xp->updateImage();
	
	addImage(Xp);
	
}

Ft2Widget::~Ft2Widget()
{}

