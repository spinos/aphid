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
#include "dtdwt1.h"

using namespace aphid;

DtFtWidget::DtFtWidget(QWidget *parent) : Plot1DWidget(parent)
{
	UniformPlot1D * xp = new UniformPlot1D;
	xp->create(256);
	int i=0, j;
	for(;i<256;++i) {
#if 1
		float slope = (float)(i - 128) / 128.f;
		slope = Absolute<float>(slope);
		xp->y()[i] = RandomFn11() * 0.17f 
					+ (cos(.031f * 3.14f * i) + 0.5f * cos(.053f * 3.14f * i) )
					* .71f * (1.f - slope);
#else
		xp->y()[i] = RandomF01();
#endif
	}
		
	xp->setColor(0.5,0.5,.5);
	addVectorPlot(xp);
	
	wla::DualTree tree;
	tree.analize(xp->data(), 3);
	
	const float fltcols[8][3] = {
		{1,0,0}, {0,0,1},
		{1,1,0}, {0,1,1},
		{.5,1,0}, {0,1,.5},
		{1,.5,0}, {0,.5,1}
	};
	
	for(j=0;j<=tree.numStages();++j) {
	
		for(int i=0;i<2;++i) {
			const VectorN<float> & src = tree.stage(j,i);
			
			std::cout<<"\n stage "<<j<<" w"<<i
					<<" n "<<src.N();
			
			UniformPlot1D * stp = new UniformPlot1D;
			stp->create(src.v(), src.N() );
			
			int ic = (j*2+i)&7;
			stp->setColor(fltcols[ic][0], fltcols[ic][1], fltcols[ic][2]);
			addVectorPlot(stp);
		}
	}
	
#if 0
	
	float *aft[4];
	float *sft[4];

	for(i=0;i<4;++i) {
		aft[i] = new float[10];
		sft[i] = new float[10];
	}
	
	//wla::fsfarras(aft, sft);
	wla::dualflt(aft, sft);
	
	for(i=0;i<4;++i) {
		UniformPlot1D * afp = new UniformPlot1D;
		afp->create(sft[i], 10);
		afp->setColor(fltcols[i][0], fltcols[i][1], fltcols[i][2]);
		addVectorPlot(afp);
	}
#endif
	
}

DtFtWidget::~DtFtWidget()
{}

