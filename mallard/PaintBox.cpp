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
#include <QColorEditSlider.h>

PaintBox::PaintBox(QWidget *parent) : QGroupBox(parent)
{
	QPixmap *pix = new QPixmap(":brushActive.png");
	QLabel * img = new QLabel;
	img->setPixmap(*pix);
	m_radiusValue = new QDoubleEditSlider(tr("Radius"), this);
	m_radiusValue->setLimit(0.01, 1000.0);
	m_radiusValue->setValue(1.0);
	
	m_colorValue = new QColorEditSlider(tr("Color"), this);
	
	QVBoxLayout * controlLayout = new QVBoxLayout;
	controlLayout->addWidget(img);
	controlLayout->addWidget(m_radiusValue);
	controlLayout->addWidget(m_colorValue);
	controlLayout->addStretch();
	setLayout(controlLayout);
	setTitle(tr("Paint Attribute Map"));
	
	connect(m_radiusValue, SIGNAL(valueChanged(double)), this, SLOT(sendRadius(double)));
	connect(m_colorValue, SIGNAL(valueChanged(QColor)), this, SLOT(sendColor(QColor)));
}

double PaintBox::radius() const
{
	return m_radiusValue->value();
}

QColor PaintBox::color() const { return m_colorValue->value(); }

void PaintBox::sendRadius(double x)
{
	emit radiusChanged(x);
}

void PaintBox::sendColor(QColor c) { emit colorChanged(c); }
