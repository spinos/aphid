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

namespace aphid {

QPolarCoordinateEdit::QPolarCoordinateEdit(const QString & name, QWidget *parent)
	: QWidget(parent)
{
	m_name = new QLabel(name, this);
	m_phi = new QAngleEdit(tr("Phi"), this);
	m_theta = new QAngleEdit(tr("Theta"), this);
	m_theta->setMin(-1.57);
	m_theta->setMax(1.57);
	
	QHBoxLayout *layout = new QHBoxLayout;
	layout->addWidget(m_name);
	layout->addWidget(m_phi);
	layout->addWidget(m_theta);
	layout->setStretch(0, 1);
	layout->setSpacing(0);
	
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

}
//:~