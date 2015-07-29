/*
 *  QPolarCoordinateEdit.cpp
 *  
 *
 *  Created by jian zhang on 7/30/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "QPolarCoordinateEdit.h"
#include "QAngleEdit.h"

QPolarCoordinateEdit::QPolarCoordinateEdit(QWidget *parent)
	: QWidget(parent)
{
	m_phi = new QAngleEdit(this);
	m_theta = new QAngleEdit(this);
	m_theta->setMin(-1.57);
	m_theta->setMax(1.57);
	
	QHBoxLayout *layout = new QHBoxLayout;
	layout->addWidget(m_phi);
	layout->addWidget(m_theta);
	layout->addStretch(1);
	layout->setSpacing(2);
	
	setLayout(layout);
	
	connect(m_phi, SIGNAL(valueChanged(double)), this, SLOT(sendPhi(double)));
	connect(m_theta, SIGNAL(valueChanged(double)), this, SLOT(sendTheta(double)));
}
	
void QPolarCoordinateEdit::setPhi(double x)
{ m_phi->setValue(x); }

void QPolarCoordinateEdit::setTheta(double x)
{ m_theta->setValue(x); }

void QPolarCoordinateEdit::sendPhi(double x)
{ emit valueChanged(QPointF(x, m_theta->value())); }

void QPolarCoordinateEdit::sendTheta(double x)
{ emit valueChanged(QPointF(m_phi->value(), x)); }
//:~