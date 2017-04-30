/*
 *  CombBox.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/23/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "CombBox.h"

#include <QtGui>
#include <QDoubleEditSlider.h>

CombBox::CombBox(QWidget *parent) : QGroupBox(parent)
{
	QPixmap *pix = new QPixmap(":combActive.png");
	QLabel * img = new QLabel;
	img->setPixmap(*pix);
	m_radiusValue = new QDoubleEditSlider(tr("Radius"));
	m_radiusValue->setLimit(0.01, 1000.0);
	m_radiusValue->setValue(1.0);
	QVBoxLayout * controlLayout = new QVBoxLayout;
	controlLayout->addWidget(img);
	controlLayout->addWidget(m_radiusValue);
	controlLayout->addStretch();
	setLayout(controlLayout);
	setTitle(tr("Change Direction"));
	
	connect(m_radiusValue, SIGNAL(valueChanged(double)), this, SLOT(sendRadius(double)));
}

double CombBox::radius() const
{
	return m_radiusValue->value();
}

void CombBox::sendRadius(double x)
{
	emit radiusChanged(x);
}