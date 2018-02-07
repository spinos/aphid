/*
 *  linregwidget.cpp
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "LinregWidget.h"
#include <math/miscfuncs.h>
#include <math/LeastMeanSquares.h>
#include <math/linearMath.h>

using namespace aphid;

LinregWidget::LinregWidget(QWidget *parent) : Plot1DWidget(parent)
{
	setBound(0, 100, 10, 0, 1, 4);
	
#define SEQ_LEN 100
	
	m_trainPlot = new UniformPlot1D;
	m_trainPlot->create(SEQ_LEN);
	int i=0;
	for(;i<SEQ_LEN;++i) {
		m_trainPlot->x()[i] = i;
		m_trainPlot->y()[i] = 
			0.3f + .003f * i + .05f * sin(.49f * i - .2f) + .1f* cos(.29f * i + .32f ) + RandomFn11() * .1f;
	}
	
	m_trainPlot->setColor(0,0,1);
	m_trainPlot->setGeomType(UniformPlot1D::GtMark);
	addVectorPlot(m_trainPlot);
	
/// filter data
	DenseVector<float> xf(7);
	xf.setZero();
	DenseVector<float> hf(7);
	hf.setZero();
	
	LeastMeanSquares<float, 7> estimator;
	estimator.setData(xf.v(), hf.v() );

	m_predictPlot = new UniformPlot1D;
	m_predictPlot->create(SEQ_LEN);
	
	m_blendPlot = new UniformPlot1D;
	m_blendPlot->create(SEQ_LEN);
	
	for(i=0;i<SEQ_LEN;++i) {
	    
	    m_predictPlot->x()[i] = i;
		
	    float yhat = estimator.predict(m_trainPlot->y()[i], i);
	    m_predictPlot->y()[i] = yhat;
		
		std::cout//<<"\n x("<<i<<") "<<xf;
			<<"\n h("<<i<<") "<<hf;
			
	}
	m_predictPlot->setColor(1,.67,0);
	addVectorPlot(m_predictPlot);
	
	for(i=0;i<SEQ_LEN;++i) {
	    
	    m_blendPlot->x()[i] = i;
	    float a = 1.f - 1.f / (1.f + i);
		
		if(i< 1)
		    m_blendPlot->y()[i] = m_trainPlot->y()[i];
		else
		    m_blendPlot->y()[i] = (1.f - a) * m_trainPlot->y()[i] + a * m_blendPlot->y()[i-1];
	}
	m_blendPlot->setColor(.3,1,.5);
	addVectorPlot(m_blendPlot);
}

LinregWidget::~LinregWidget()
{}

