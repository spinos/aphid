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
	int i=0;
	for(;i<dim;++i)
		ap->y()[i] = x.column(i)[0];
		
	ap->setColor(0,0,1);
	addVectorPlot(ap);
}

SampleKernelWidget::~SampleKernelWidget()
{}

}
}

