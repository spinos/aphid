/*
 *  RedBlueControl.cpp
 *  cudafem
 *
 *  Created by jian zhang on 9/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include <QDoubleEditSlider.h>
#include "RedBlueControl.h"

namespace ttg {

RedBlueControl::RedBlueControl(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("Red Blue Refine Control"));
    
    m_aValue = new QDoubleEditSlider(tr("a"), this);
	m_aValue->setLimit(-1.0, 1.0);
	m_aValue->setValue(1.0);
	
	m_bValue = new QDoubleEditSlider(tr("b"), this);
	m_bValue->setLimit(-1.0, 1.0);
	m_bValue->setValue(1.0);
	
	m_cValue = new QDoubleEditSlider(tr("c"), this);
	m_cValue->setLimit(-1.0, 1.0);
	m_cValue->setValue(1.0);
	
	m_dValue = new QDoubleEditSlider(tr("d"), this);
	m_dValue->setLimit(-1.0, 1.0);
	m_dValue->setValue(1.0);
	
	S1Grp = new QGroupBox;
    QVBoxLayout * yLayout = new QVBoxLayout;
	yLayout->addWidget(m_aValue);
	yLayout->addWidget(m_bValue);
	yLayout->addWidget(m_cValue);
	yLayout->addWidget(m_dValue);
	yLayout->setStretch(4, 1);
	
    S1Grp->setLayout(yLayout);
	
    QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(S1Grp);
	layout->setStretch(2, 1);
	layout->setSpacing(4);
	
	setLayout(layout);
    
    connect(m_aValue, SIGNAL(valueChanged(double)), this, SLOT(sendA(double)));
    connect(m_bValue, SIGNAL(valueChanged(double)), this, SLOT(sendB(double)));
    connect(m_cValue, SIGNAL(valueChanged(double)), this, SLOT(sendC(double)));
    connect(m_dValue, SIGNAL(valueChanged(double)), this, SLOT(sendD(double)));
    
}

void RedBlueControl::sendA(double x)
{ emit aChanged(x); }

void RedBlueControl::sendB(double x)
{ emit bChanged(x); }

void RedBlueControl::sendC(double x)
{ emit cChanged(x); }

void RedBlueControl::sendD(double x)
{ emit dChanged(x); }

}
//:~