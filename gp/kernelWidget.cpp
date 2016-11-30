/*
 *  kernelWidget.cpp
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "kernelWidget.h"
#include <linearMath.h>
using namespace lfr;

namespace aphid {
namespace gpr {

KernelWidget::KernelWidget(QWidget *parent) : Plot2DWidget(parent)
{}

KernelWidget::~KernelWidget()
{}

void KernelWidget::plotK(const DenseMatrix<float> * K)
{
    std::cout<<" plot K ";//<<*K;
    UniformPlot2DImage * Kp = new UniformPlot2DImage;
	Kp->create(K->numRows(), K->numColumns(), 3);
	Kp->setDrawScale(255.f/(float)K->numRows() );
	Kp->floatToColor(K->column(0) );
	Kp->updateImage();
	addImage(Kp);
}

}
}
