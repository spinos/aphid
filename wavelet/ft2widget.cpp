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
#include <ANoise3.h>

using namespace aphid;

Ft2Widget::Ft2Widget(QWidget *parent) : Plot2DWidget(parent)
{
	UniformPlot2DImage * Xp = new UniformPlot2DImage;
#define DIM_Y 128
#define DIM_X 128
#define DIM_Z 1
	Xp->create(DIM_Y, DIM_X, DIM_Z);
	
	float d = .012352f;
	
	int i, j, k;
	for(k=0;k<DIM_Z;++k) {
		float * Xv = Xp->y(k);
		
		Vector3F o(0.435f + k, 0.73656f + k, 0.3765f + k);
		
	for(j=0;j<DIM_X;++j) {
		for(i=0;i<DIM_Y;++i) {
#if 1
			Vector3F p(d*(i-31), d*(j-31), d*(k-31) );
			float r = ANoise3::Fbm((const float *)&p,
											(const float *)&o,
											3.23f,
											11,
											1.4141f,
											.853f);
			Clamp01(r);
			r *= .9f;
#else
			float r = RandomF01();
#endif
			Xv[j*DIM_Y+i] = r;
		}
	}
	}
	
	Xp->setDrawScale(1.f);
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
	
	lowpassP->setDrawScale(1.f);
	lowpassP->updateImage();
	
	addImage(lowpassP);
	
	highpassP->setDrawScale(1.f);
	highpassP->updateImage(true);
	
	addImage(highpassP);
}

Ft2Widget::~Ft2Widget()
{}

