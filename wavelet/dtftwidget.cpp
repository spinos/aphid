/*
 *  dtftwidget.cpp
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include <Calculus.h>
#include "DtFtWidget.h"
#include "fltbanks.h"

using namespace aphid;

DtFtWidget::DtFtWidget(QWidget *parent) : Plot1DWidget(parent)
{
	UniformPlot1D * ap = new UniformPlot1D;
	ap->create(64);
	int i=0;
	for(;i<64;++i)
		ap->y()[i] = RandomF01();
		
	ap->setColor(0,0,1);
	//addVectorPlot(ap);
	
	float *aft[4];
	float *sft[4];

	for(i=0;i<4;++i) {
		aft[i] = new float[10];
		sft[i] = new float[10];
	}
	
	wla::fsfarras(aft, sft);
	
	const float fltcols[4][3] = {
		{1,0,0}, {0,0,1},
		{1,1,0}, {0,1,1}
	};
	
	for(i=0;i<4;++i) {
		UniformPlot1D * afp = new UniformPlot1D;
		afp->create(aft[i], 10);
		afp->setColor(fltcols[i][0], fltcols[i][1], fltcols[i][2]);
		addVectorPlot(afp);
	}
}

DtFtWidget::~DtFtWidget()
{}

