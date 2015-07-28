/*
 *  PhysicsControl.cpp
 *  cudafem
 *
 *  Created by jian zhang on 9/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include <QIntEditSlider.h>
#include <QDoubleEditSlider.h>
#include "PhysicsControl.h"

PhysicsControl::PhysicsControl(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("Physics Control"));
    YGrp = new QGroupBox;
    
    m_youngModulusValue = new QDoubleEditSlider(tr("Young's modulus"), this);
	m_youngModulusValue->setLimit(40000.0, 800000.0);
	m_youngModulusValue->setValue(160000.0);
    
    QHBoxLayout * yLayout = new QHBoxLayout;
	yLayout->addWidget(m_youngModulusValue);
	yLayout->setStretch(1, 1);
	
    YGrp->setLayout(yLayout);
    
    QHBoxLayout *layout = new QHBoxLayout;
	
	layout->addWidget(YGrp);
	layout->setStretch(1, 1);
	layout->setSpacing(4);
	
	setLayout(layout);
    
    connect(m_youngModulusValue, SIGNAL(valueChanged(double)), this, SLOT(sendYoungModulus(double)));
}

void PhysicsControl::sendYoungModulus(double x)
{ emit youngsModulusChanged(x); }
//:~
