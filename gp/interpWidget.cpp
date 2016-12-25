/*
 *  interpWidget.cpp
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "interpWidget.h"
#include <gpr/Interpolate1D.h>
#include <math/linearMath.h>

using namespace aphid;

InterpWidget::InterpWidget(QWidget *parent) : Plot1DWidget(parent),
m_selectedTrainInd(-1)
{
#if 1
	std::cout<<"\n test transMult";
	
	static const float vK[3*3] = {
		2,0,0,
		0,2,0,
		0,0,2
	};
	DenseMatrix<float> K(3,3); K.copyData(vK);
	
	static const float vY[3] = {1,2,3};
	DenseMatrix<float> Y(3,1); Y.copyData(vY);
	
	DenseMatrix<float> Yt = Y.transposed();
	
	DenseMatrix<float> YtK(1,3);
	Y.transMult(YtK, K);
	
	DenseMatrix<float> YtKY(1,1);
	YtK.mult(YtKY, Y);
	
	std::cout<<"\n Y = "<<Y
			<<"\n Y**T = "<<Yt
			<<"\n K = "<<K
			<<"\n Y**T * K = "<<YtK
			<<"\n Y**T * K * Y= "<<YtKY;
	
#endif

#define DIM_TRAIN 9
static const float initTrain[DIM_TRAIN][2] = {
    {-1.5f, -1.f},
    {-.75f, -.1f},
	{-.25f, .4f},
    { .0f, 1.5f},
	{ .3f, 1.45f},
    { .5f, 1.75f},
    { .67f,  .6f},
	{ 1.5f,  .1f},
	{ 1.75f,  -.4f}
};

    setBound(-2, 2, 4, -3, 3, 4);
    
    m_gpi = new gpr::Interpolate1D();
	m_gpi->setBound(-2.f, 2.f);
	
	m_trainPlot = new UniformPlot1D;
	m_trainPlot->create(DIM_TRAIN);
	int i=0;
	for(;i<DIM_TRAIN;++i) {
		m_trainPlot->x()[i] = initTrain[i][0];
		m_trainPlot->y()[i] = initTrain[i][1];
		m_gpi->addObservation(initTrain[i][0], initTrain[i][1]);
	}
		
	m_trainPlot->setColor(0,0,1);
	m_trainPlot->setGeomType(UniformPlot1D::GtMark);
	addVectorPlot(m_trainPlot);
	
	if(!m_gpi->learn() ) {
	     throw "gp failed to learn";   
	}
	
	const int predDim = 50;
	const float oox = 1.f / (predDim-1);
	
	m_predictPlot = new UniformPlot1D;
	m_predictPlot->create(predDim);
	
	for(i=0;i<predDim;++i) {
		m_predictPlot->y()[i] = m_gpi->predict(xToBound(oox * i) );
	}
		
	m_predictPlot->setColor(0,.8f,.1f);
	addVectorPlot(m_predictPlot);
	
}

InterpWidget::~InterpWidget()
{}

void InterpWidget::mousePressEvent(QMouseEvent *event)
{
	Vector2F vmouse = toRealSpace(event->x(), event->y());
	const int & n = m_trainPlot->numY();
	for(int i=0;i<n;++i) {
	    Vector2F vtest(m_trainPlot->x()[i], m_trainPlot->y()[i]);
	    if(vtest.distanceTo(vmouse) < 0.1f ) {
	        m_selectedTrainInd = i;
	        break;
	    }
	}
}

void InterpWidget::mouseMoveEvent(QMouseEvent *event)
{
    if(m_selectedTrainInd < 0) {
        return;   
    }
    
	Vector2F vmouse = toRealSpace(event->x(), event->y());
	m_trainPlot->x()[m_selectedTrainInd] = vmouse.x;
	m_trainPlot->y()[m_selectedTrainInd] = vmouse.y;
	
	m_gpi->setObservation(vmouse.x, vmouse.y, m_selectedTrainInd);
	m_gpi->learn();
	const int predDim = m_predictPlot->numY();
	const float oox = 1.f / (predDim-1);
	
	for(int i=0;i<predDim;++i) {
		m_predictPlot->y()[i] = m_gpi->predict(xToBound(oox * i) );
	}
	update();
}

void InterpWidget::mouseReleaseEvent(QMouseEvent *event)
{ 
	m_selectedTrainInd = -1;
}
