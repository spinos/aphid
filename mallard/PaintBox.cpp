/*
 *  PaintBox.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/26/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "PaintBox.h"


#include <QtGui>
#include <QDoubleEditSlider.h>

PaintBox::PaintBox(QWidget *parent) : QGroupBox(parent)
{
	QPixmap *pix = new QPixmap(":brushActive.png");
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
	setTitle(tr("Paint Attribute Map"));
	
	connect(m_radiusValue, SIGNAL(valueChanged(double)), this, SLOT(sendRadius(double)));
}

double PaintBox::radius() const
{
	return m_radiusValue->value();
}

void PaintBox::sendRadius(double x)
{
	emit radiusChanged(x);
}