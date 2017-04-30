/*
 *  ScaleBox.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/23/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "ScaleBox.h"

#include <QtGui>
#include <QDoubleEditSlider.h>

ScaleBox::ScaleBox(QWidget *parent) : QGroupBox(parent)
{
	QPixmap *pix = new QPixmap(":featherLengthActive.png");
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
	setTitle(tr("Scale Length"));
	
	connect(m_radiusValue, SIGNAL(valueChanged(double)), this, SLOT(sendRadius(double)));
}

double ScaleBox::radius() const
{
	return m_radiusValue->value();
}

void ScaleBox::sendRadius(double x)
{
	emit radiusChanged(x);
}