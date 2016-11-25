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
#include "RbfKernel.h"
#include "Covariance.h"
#include "linspace.h"

using namespace lfr;

namespace aphid {
namespace gpr {

SampleKernelWidget::SampleKernelWidget(const DenseMatrix<float> & yActual,
						QWidget *parent) : Plot1DWidget(parent)
{
	int dim = yActual.numColumns();
	
	setBound(-1, 1, 4, -3, 3, 4);
	
	UniformPlot1D * ap = new UniformPlot1D;
	ap->create(dim);
	int i=0, j;
	for(;i<dim;++i)
		ap->y()[i] = yActual.column(i)[0];
		
	ap->setColor(0,0,1);
	addVectorPlot(ap);
	
	int ydim = 0;
	for(i=2;i<dim;i+=4) {
		ydim++;
	}
	
	std::cout<<"\n train dim "<<ydim;
	
	DenseMatrix<float> xTrain(ydim, 1);
	DenseMatrix<float> yTrain(ydim, 1);
	
	const float oox = 1.f / (dim-1);
	
	for(i=2,j=0;i<dim;i+=4) {
		xTrain.column(0)[j] = -1.f + 2.f * oox*i;
		yTrain.column(0)[j++] = ap->y()[i];
	}
	
	std::cout<<"\n xTrain"<<xTrain;
	std::cout<<"\n yTrain"<<yTrain;
	
	UniformPlot1D * plTrain = new UniformPlot1D;
	plTrain->create(ydim);
	
	for(i=0;i<ydim;++i) {
		plTrain->x()[i] = xTrain.column(0)[i];
		plTrain->y()[i] = yTrain.column(0)[i];
	}
		
	plTrain->setGeomType(UniformPlot1D::GtMark);
	plTrain->setColor(0,.5,1);
	addVectorPlot(plTrain);
	
	std::cout<<"\n build train kernel";
    RbfKernel<float> rbf(0.33);
    
	Covariance<float, RbfKernel<float> > covTrain;
    covTrain.create(xTrain, rbf);
	
	std::cout<<"\n Ktrain"<<covTrain.K();
	
	std::cout<<"\n build test x";
	DenseMatrix<float> xTest(100,1);
    linspace<float>(xTest.column(0), -1.f, 1.f, 100);
    
	Covariance<float, RbfKernel<float> > covTest;
    covTest.create(xTest, xTrain, rbf);
	
	std::cout.flush();
	
}

SampleKernelWidget::~SampleKernelWidget()
{}

}
}

