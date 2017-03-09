/*
 *  ft1dWidget.cpp
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include <math/Calculus.h>
#include "ft1dWidget.h"
#include "fltbanks.h"

using namespace aphid;

Ft1dWidget::Ft1dWidget(QWidget *parent) : Plot1DWidget(parent)
{
	UniformPlot1D * ap = new UniformPlot1D;
	ap->create(64);
	int i=0;
	for(;i<64;++i)
		ap->y()[i] = RandomF01();
		
	ap->setColor(0,0,1);
	addVectorPlot(ap);

	float * Xshf = wla::circshift(ap->y(), 64, -5);
	
	float * locoeff;
	float * hicoeff;
	int ncoeff;
	wla::afb(Xshf, 64, locoeff, hicoeff, ncoeff);
	
	UniformPlot1D * flo = new UniformPlot1D;
	flo->create(locoeff, ncoeff);
	flo->setColor(1.,1.,0.);
	addVectorPlot(flo);
	
	UniformPlot1D * fhi = new UniformPlot1D;
	fhi->create(hicoeff, ncoeff);
	fhi->setColor(0.,1.,1.);
	addVectorPlot(fhi);
	
	float * ysyn;
	int nysyn;
	wla::sfb(locoeff, hicoeff, ncoeff, ysyn, nysyn);
	
	std::cout<<"\n max abs err "<<calc::maxAbsoluteError(ap->y(), ysyn, 64);
	std::cout.flush();
	
	UniformPlot1D * fysn = new UniformPlot1D;
	fysn->create(ysyn, nysyn);
	fysn->setColor(1.,0.,0.);
	fysn->setLineStyle(UniformPlot1D::LsDash);
	addVectorPlot(fysn);
	
	delete[] Xshf;
	delete[] locoeff;
	delete[] hicoeff;
	delete[] ysyn;
	
	float *aft[2];
	aft[0] = new float[10];
	aft[1] = new float[10];
	float *sft[2];
	sft[0] = new float[10];
	sft[1] = new float[10];
	wla::farras(aft, sft);
	
	UniformPlot1D * paf = new UniformPlot1D;
	paf->create(aft[0], 10);
	paf->setColor(1.,0., 0.);
	addVectorPlot(paf);
	
	UniformPlot1D * psf = new UniformPlot1D;
	psf->create(aft[1], 10);
	psf->setColor(0.,1., 0.);
	addVectorPlot(psf);
	
	delete[] aft[0];
	delete[] aft[1];
	delete[] sft[0];
	delete[] sft[1];
}

Ft1dWidget::~Ft1dWidget()
{}

