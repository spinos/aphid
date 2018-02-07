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
#include <math/LinearRegression.h>

using namespace aphid;

LinregWidget::LinregWidget(QWidget *parent) : Plot1DWidget(parent)
{
	setBound(0, 1, 10, 0, 1, 4);
	
#define SEQ_LEN 40
#define Dx .025f
	
	m_trainPlot = new UniformPlot1D;
	m_trainPlot->create(SEQ_LEN);
	int i=0;
	for(;i<SEQ_LEN;++i) {
		m_trainPlot->x()[i] = Dx * i;
		m_trainPlot->y()[i] = .5f + (.13f - Dx * i * .05f) * cos(3.14f * .43f * i + RandomFn11() * .19f ) + RandomFn11() * .07f;
	}
	
	m_trainPlot->setColor(0,0,1);
	m_trainPlot->setGeomType(UniformPlot1D::GtMark);
	addVectorPlot(m_trainPlot);
	
	LinearRegressionData<float, 4> model;
	LinearRegressionPredictor<float, 4> estimator;
	estimator.setData(&model);

	m_predictPlot = new UniformPlot1D;
	m_predictPlot->create(SEQ_LEN);
	
	m_blendPlot = new UniformPlot1D;
	m_blendPlot->create(SEQ_LEN);
	
	for(i=0;i<SEQ_LEN;++i) {
	    
	    m_predictPlot->x()[i] = Dx * i;
	    float yhat = estimator.updateAndPredict(m_trainPlot->y()[i], i);
	    m_predictPlot->y()[i] = yhat;
	    
	}
	m_predictPlot->setColor(1,.7,.1);
	addVectorPlot(m_predictPlot);
	
	for(i=0;i<SEQ_LEN;++i) {
	    
	    m_blendPlot->x()[i] = Dx * i;
	    float a = 1.f - 3.f / (3.f + i);
		
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

