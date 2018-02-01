/*
 *  haltonwidget.cpp
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "haltonwidget.h"

using namespace aphid;

HaltonWidget::HaltonWidget(QWidget *parent) : Plot1DWidget(parent)
{
	setBound(0, 1, 4, 0, 1, 4);
	
#define SEQ_LEN 256
	
	m_trainPlot = new UniformPlot1D;
	m_trainPlot->create(SEQ_LEN);
	int i=0;
	for(;i<SEQ_LEN;++i) {
		m_trainPlot->x()[i] = calcHalton(i+1, 2);
		m_trainPlot->y()[i] = calcHalton(i+1, 3);
	}
	
	m_trainPlot->setColor(0,0,1);
	m_trainPlot->setGeomType(UniformPlot1D::GtMark);
	addVectorPlot(m_trainPlot);
#if 0	
	std::cout<<" print halton(2,3) sequence 128 {";
	for(i=0;i<SEQ_LEN;++i) {
		std::cout<<"{ "<<m_trainPlot->x()[i]
				<<"f, "<<m_trainPlot->y()[i]<<"f }, ";
	}
	std::cout<<"}";
	std::cout.flush();
#endif
}

HaltonWidget::~HaltonWidget()
{}

float HaltonWidget::calcHalton(int i, int b)
{
	float f = 1.f;
	float r = 0.f;
	
	while(i > 0) {
		f /= (float)b;
		r += f * (i % b);
		i /= b;
	}
	return r;
}
