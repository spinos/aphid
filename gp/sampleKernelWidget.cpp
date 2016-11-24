/*
 *  sampleKernelWidget.cpp
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "sampleKernelWidget.h"

namespace aphid {
namespace gpr {

SampleKernelWidget::SampleKernelWidget(const lfr::DenseMatrix<float> & x,
						QWidget *parent) : Plot1DWidget(parent)
{
	int dim = x.numColumns();
	
	setBound(0, dim, 5, -3, 3, 6);
	
	UniformPlot1D * ap = new UniformPlot1D;
	ap->create(dim);
	int i=0, j;
	for(;i<dim;++i)
		ap->y()[i] = x.column(i)[0];
		
	ap->setColor(0,0,1);
	addVectorPlot(ap);
	
	int ydim = 0;
	for(i=2;i<dim;i+=4) {
		ydim++;
	}
	
	std::cout<<"\n dim y "<<ydim;
	
	UniformPlot1D * yp = new UniformPlot1D;
	yp->create(ydim);
	
	float oox = 1.f / (dim-1);
	
	for(i=2,j=0;i<dim;i+=4) {
		yp->x()[j] = oox*i;
		yp->y()[j++] = x.column(i)[0];
	}
		
	yp->setGeomType(UniformPlot1D::GtMark);
	yp->setColor(0,1,1);
	addVectorPlot(yp);
	
	std::cout.flush();
	
}

SampleKernelWidget::~SampleKernelWidget()
{}

}
}

