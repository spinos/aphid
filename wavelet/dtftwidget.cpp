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
#include "dtftwidget.h"
#include "dtdwt1.h"
#include "gensig.h"

using namespace aphid;

DtFtWidget::DtFtWidget(QWidget *parent) : Plot1DWidget(parent)
{
	UniformPlot1D * xp = new UniformPlot1D;
	gen1dsig(xp, 256);
		
	xp->setColor(0.,0.,0.);
	addVectorPlot(xp);
	
	wla::DualTree tree;
	tree.analize(xp->data(), 3);
	
	const float fltcols[8][3] = {
		{1,0,0}, {0,0,1},
		{1,1,0}, {0,1,1},
		{.5,1,0}, {0,1,.5},
		{1,.5,0}, {0,.5,1}
	};
	
	int i, j;
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
	
	VectorN<float> result;
	tree.synthesize(result);
	
	float mxe = 0.f;
	result.maxAbsError(mxe, xp->data() );
	std::cout<<"\n sythesized n "<<result.N()<<" max err "<<mxe;
	
	UniformPlot1D * yp = new UniformPlot1D;
	yp->create(result.v(), result.N() );
	yp->setColor(1.,0.,.75);
	yp->setLineStyle(UniformPlot1D::LsDash);
	addVectorPlot(yp);
	
#if 0
	
	float *aft[4];
	float *sft[4];

	for(i=0;i<4;++i) {
		aft[i] = new float[10];
		sft[i] = new float[10];
	}
	
	//wla::Dtf::fsfarras(aft, sft);
	wla::Dtf::dualflt(aft, sft);
	
	for(i=0;i<4;++i) {
		UniformPlot1D * afp = new UniformPlot1D;
		afp->create(sft[i], 10);
		afp->setColor(fltcols[i][0], fltcols[i][1], fltcols[i][2]);
		addVectorPlot(afp);
	}
#endif
	std::cout.flush();
	
}

DtFtWidget::~DtFtWidget()
{}

