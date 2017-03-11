/*
 *  SynthControl.cpp
 *  cudafem
 *
 *  Created by jian zhang on 9/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include <qt/QDoubleEditSlider.h>
#include <img/ExrImage.h>
#include <qt/NavigatorWidget.h>
#include "SynthControl.h"

using namespace aphid;

SynthControl::SynthControl(const ExrImage * img, QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("Synthesis Control"));
	
	m_navigator = new NavigatorWidget(img, this);
    
    m_aValue = new QDoubleEditSlider(tr("L0 scale"), this);
	m_aValue->setLimit(0.0, 1.0);
	m_aValue->setValue(1.0);
	
	m_bValue = new QDoubleEditSlider(tr("L1 scale"), this);
	m_bValue->setLimit(0.0, 1.0);
	m_bValue->setValue(1.0);
	
	m_cValue = new QDoubleEditSlider(tr("L2 scale"), this);
	m_cValue->setLimit(0.0, 1.0);
	m_cValue->setValue(1.0);
	
	m_dValue = new QDoubleEditSlider(tr("L3 scale"), this);
	m_dValue->setLimit(0.0, 1.0);
	m_dValue->setValue(1.0);
	
	S1Grp = new QGroupBox;
    QVBoxLayout * yLayout = new QVBoxLayout;
	yLayout->addWidget(m_navigator);
	yLayout->addWidget(m_aValue);
	yLayout->addWidget(m_bValue);
	yLayout->addWidget(m_cValue);
	yLayout->addWidget(m_dValue);
    S1Grp->setLayout(yLayout);
	
    QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(S1Grp);
	layout->addStretch();
	layout->setSpacing(4);
	
	setLayout(layout);
    
    connect(m_aValue, SIGNAL(valueChanged(double)), this, SLOT(sendA(double)));
    connect(m_bValue, SIGNAL(valueChanged(double)), this, SLOT(sendB(double)));
    connect(m_cValue, SIGNAL(valueChanged(double)), this, SLOT(sendC(double)));
    connect(m_dValue, SIGNAL(valueChanged(double)), this, SLOT(sendD(double)));
    
}

void SynthControl::sendA(double x)
{ emit l0ScaleChanged(x); }

void SynthControl::sendB(double x)
{ emit l1ScaleChanged(x); }

void SynthControl::sendC(double x)
{ emit l2ScaleChanged(x); }

void SynthControl::sendD(double x)
{ emit l3ScaleChanged(x); }

//:~