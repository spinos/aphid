/*
 *  paramwidget.cpp
 *  
 *
 *  Created by jian zhang on 9/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "paramwidget.h"
#include <qt/QDoubleEditSlider.h>

using namespace aphid;

ParamWidget::ParamWidget(QWidget *parent) : QWidget(parent)
{
	m_camberEdit = new QDoubleEditSlider("camber (m)", this);
	m_camberEdit->setLimit(0.0, 50.0);
	m_camberEdit->setValue(2.0);
	
	m_positionEdit = new QDoubleEditSlider("postion (p)", this);
	m_positionEdit->setLimit(0.0, 1.0);
	m_positionEdit->setValue(0.4);
	
	m_thicknessEdit = new QDoubleEditSlider("thickness (t)", this);
	m_thicknessEdit->setLimit(0.0, 1.0);
	m_thicknessEdit->setValue(0.15);
	
	
	QVBoxLayout * yLayout = new QVBoxLayout;
	yLayout->addWidget(m_camberEdit);
	yLayout->addWidget(m_positionEdit);
	yLayout->addWidget(m_thicknessEdit);
	yLayout->addStretch(3);
	
	setLayout(yLayout);
	
	connect(m_camberEdit, SIGNAL(valueChanged(double)), this, SLOT(sendCamber(double)));
    connect(m_positionEdit, SIGNAL(valueChanged(double)), this, SLOT(sendPosition(double)));
    connect(m_thicknessEdit, SIGNAL(valueChanged(double)), this, SLOT(sendThickness(double)));
    
}

ParamWidget::~ParamWidget()
{}

void ParamWidget::sendCamber(double x)
{ emit camberChanged(x); }

void ParamWidget::sendPosition(double x)
{ emit positionChanged(x); }
	
void ParamWidget::sendThickness(double x)
{ emit thicknessChanged(x); }
