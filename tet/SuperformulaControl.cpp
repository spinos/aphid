/*
 *  SuperformulaControl.cpp
 *  cudafem
 *
 *  Created by jian zhang on 9/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include <QDoubleEditSlider.h>
#include "SuperformulaControl.h"

namespace ttg {

SuperformulaControl::SuperformulaControl(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("Superformula Control"));
    
    m_a1Value = new QDoubleEditSlider(tr("a"), this);
	m_a1Value->setLimit(0.1, 10);
	m_a1Value->setValue(1.0);
	
	m_b1Value = new QDoubleEditSlider(tr("b"), this);
	m_b1Value->setLimit(0.1, 10);
	m_b1Value->setValue(1.0);
	
	m_m1Value = new QDoubleEditSlider(tr("m"), this);
	m_m1Value->setLimit(-30, 30);
	m_m1Value->setValue(4.0);
	
	m_n1Value = new QDoubleEditSlider(tr("n1"), this);
	m_n1Value->setLimit(-90, 90);
	m_n1Value->setValue(10.0);
	
	m_n2Value = new QDoubleEditSlider(tr("n2"), this);
	m_n2Value->setLimit(-90, 90);
	m_n2Value->setValue(10.0);
	
	m_n3Value = new QDoubleEditSlider(tr("n3"), this);
	m_n3Value->setLimit(-90, 90);
	m_n3Value->setValue(10.0);
	
	S1Grp = new QGroupBox;
    QVBoxLayout * yLayout = new QVBoxLayout;
	yLayout->addWidget(m_a1Value);
	yLayout->addWidget(m_b1Value);
	yLayout->addWidget(m_m1Value);
	yLayout->addWidget(m_n1Value);
	yLayout->addWidget(m_n2Value);
	yLayout->addWidget(m_n3Value);
	yLayout->setStretch(4, 1);
	
    S1Grp->setLayout(yLayout);
	
	m_a2Value = new QDoubleEditSlider(tr("a"), this);
	m_a2Value->setLimit(0.1, 10);
	m_a2Value->setValue(1.0);
	
	m_b2Value = new QDoubleEditSlider(tr("b"), this);
	m_b2Value->setLimit(0.1, 10);
	m_b2Value->setValue(1.0);
	
	m_m2Value = new QDoubleEditSlider(tr("m"), this);
	m_m2Value->setLimit(-30, 30);
	m_m2Value->setValue(4.0);
	
	m_n21Value = new QDoubleEditSlider(tr("n1"), this);
	m_n21Value->setLimit(-90, 90);
	m_n21Value->setValue(10.0);
	
	m_n22Value = new QDoubleEditSlider(tr("n2"), this);
	m_n22Value->setLimit(-90, 90);
	m_n22Value->setValue(10.0);
	
	m_n23Value = new QDoubleEditSlider(tr("n3"), this);
	m_n23Value->setLimit(-90, 90);
	m_n23Value->setValue(10.0);
	
	S2Grp = new QGroupBox;
    QVBoxLayout * zLayout = new QVBoxLayout;
	zLayout->addWidget(m_a2Value);
	zLayout->addWidget(m_b2Value);
	zLayout->addWidget(m_m2Value);
	zLayout->addWidget(m_n21Value);
	zLayout->addWidget(m_n22Value);
	zLayout->addWidget(m_n23Value);
	zLayout->setStretch(4, 1);
	
    S2Grp->setLayout(zLayout);
	
    QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(S1Grp);
	layout->addWidget(S2Grp);
	layout->setStretch(3, 1);
	layout->setSpacing(4);
	
	setLayout(layout);
    
    connect(m_a1Value, SIGNAL(valueChanged(double)), this, SLOT(sendA1(double)));
    connect(m_b1Value, SIGNAL(valueChanged(double)), this, SLOT(sendB1(double)));
    connect(m_m1Value, SIGNAL(valueChanged(double)), this, SLOT(sendM1(double)));
    connect(m_n1Value, SIGNAL(valueChanged(double)), this, SLOT(sendN1(double)));
    connect(m_n2Value, SIGNAL(valueChanged(double)), this, SLOT(sendN2(double)));
    connect(m_n3Value, SIGNAL(valueChanged(double)), this, SLOT(sendN3(double)));
    
	connect(m_a2Value, SIGNAL(valueChanged(double)), this, SLOT(sendA2(double)));
    connect(m_b2Value, SIGNAL(valueChanged(double)), this, SLOT(sendB2(double)));
    connect(m_m2Value, SIGNAL(valueChanged(double)), this, SLOT(sendM2(double)));
    connect(m_n21Value, SIGNAL(valueChanged(double)), this, SLOT(sendN21(double)));
    connect(m_n22Value, SIGNAL(valueChanged(double)), this, SLOT(sendN22(double)));
    connect(m_n23Value, SIGNAL(valueChanged(double)), this, SLOT(sendN23(double)));
    
}

void SuperformulaControl::sendA1(double x)
{ emit a1Changed(x); }

void SuperformulaControl::sendB1(double x)
{ emit b1Changed(x); }

void SuperformulaControl::sendM1(double x)
{ emit m1Changed(x); }

void SuperformulaControl::sendN1(double x)
{ emit n1Changed(x); }

void SuperformulaControl::sendN2(double x)
{ emit n2Changed(x); }

void SuperformulaControl::sendN3(double x)
{ emit n3Changed(x); }

void SuperformulaControl::sendA2(double x)
{ emit a2Changed(x); }

void SuperformulaControl::sendB2(double x)
{ emit b2Changed(x); }

void SuperformulaControl::sendM2(double x)
{ emit m2Changed(x); }

void SuperformulaControl::sendN21(double x)
{ emit n21Changed(x); }

void SuperformulaControl::sendN22(double x)
{ emit n22Changed(x); }

void SuperformulaControl::sendN23(double x)
{ emit n23Changed(x); }

}
//:~