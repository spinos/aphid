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
	
	m_dropoffValue = new QDoubleEditSlider(tr("Dropoff"), this);
	m_dropoffValue->setLimit(0.0, 0.99);
	m_dropoffValue->setValue(0.0);
	
	m_colorValue = new QColorEditSlider(tr("Color"), this);
	
	m_strengthValue = new QDoubleEditSlider(tr("Strength"), this);
	m_strengthValue->setLimit(0.01, 1.0);
	m_strengthValue->setValue(1.0);
	
	QLabel *modeLabel = new QLabel(tr("Mode"));
	modeLabel->setMinimumWidth(100);
	m_modeCombo = new QComboBox;
	m_modeCombo->addItem(tr("Replace"));
	m_modeCombo->addItem(tr("Smooth"));
	
	QHBoxLayout * modeLayout = new QHBoxLayout;
	modeLayout->addWidget(modeLabel);
	modeLayout->addWidget(m_modeCombo);
	modeLayout->setStretch(1, 1);
	
	QVBoxLayout * controlLayout = new QVBoxLayout;
	controlLayout->addWidget(img);
	controlLayout->addLayout(modeLayout);
	controlLayout->addWidget(m_radiusValue);
	controlLayout->addWidget(m_colorValue);
	controlLayout->addWidget(m_dropoffValue);
	controlLayout->addWidget(m_strengthValue);
	controlLayout->addStretch();
	setLayout(controlLayout);
	setTitle(tr("Paint Attribute Map"));
	
	connect(m_radiusValue, SIGNAL(valueChanged(double)), this, SLOT(sendRadius(double)));
	connect(m_colorValue, SIGNAL(valueChanged(QColor)), this, SLOT(sendColor(QColor)));
	connect(m_dropoffValue, SIGNAL(valueChanged(double)), this, SLOT(sendDropoff(double)));
	connect(m_strengthValue, SIGNAL(valueChanged(double)), this, SLOT(sendStrength(double)));
	connect(m_modeCombo, SIGNAL(activated(int)), this, SLOT(sendMode(int)));
}

double PaintBox::radius() const
{
	return m_radiusValue->value();
}

QColor PaintBox::color() const { return m_colorValue->value(); }

double PaintBox::dropoff() const { return m_dropoffValue->value(); }

double PaintBox::strength() const { return m_strengthValue->value(); }

void PaintBox::sendRadius(double x) { emit radiusChanged(x); }

void PaintBox::sendColor(QColor c) { emit colorChanged(c); }

void PaintBox::sendDropoff(double x) { emit dropoffChanged(x); }

void PaintBox::sendStrength(double x) { emit strengthChanged(x); }

void PaintBox::sendMode(int x) { emit modeChanged(x); }
