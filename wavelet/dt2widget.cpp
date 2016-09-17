/*
 *  dt2widget.cpp
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include <Calculus.h>
#include "dt2widget.h"
#include "dtdwt1.h"
#include "gensig.h"

using namespace aphid;

Dt2Widget::Dt2Widget(QWidget *parent) : Plot2DWidget(parent)
{
	UniformPlot2DImage * Xp = new UniformPlot2DImage;
#define DIM_Y 256
#define DIM_X 256
#define DIM_Z 1
	gen2dsig(Xp, DIM_Y, DIM_X, DIM_Z);
	Xp->setDrawScale(1.f);
	Xp->updateImage();
	
	addImage(Xp);
			
	std::cout.flush();	
}

Dt2Widget::~Dt2Widget()
{}

