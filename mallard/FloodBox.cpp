/*
 *  FloodBox.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/23/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "FloodBox.h"

#include <QtGui>
#include <QDoubleEditSlider.h>
#include <QIntEditSlider.h>

FloodBox::FloodBox(QWidget *parent) : QGroupBox(parent)
{
	QPixmap *pix = new QPixmap(":createFeatherActive.png");
	QLabel * img = new QLabel;
	img->setPixmap(*pix);
	m_radiusValue = new QDoubleEditSlider(tr("Radius"));
	m_radiusValue->setLimit(0.01, 1000.0);
	m_radiusValue->setValue(1.0);
	
	m_numSampleValue = new QIntEditSlider(tr("Num Samples"));
	m_numSampleValue->setLimit(8, 10000);
	m_numSampleValue->setValue(99);
	
	m_strengthValue = new QDoubleEditSlider(tr("Strength"));
	m_strengthValue->setLimit(0.01, 1.0);
	m_strengthValue->setValue(1.0);
	
	m_floodAreaCheck = new QCheckBox(tr("Flood Within Selected Region"));
	
	QVBoxLayout * controlLayout = new QVBoxLayout;
	controlLayout->addWidget(img);
	controlLayout->addWidget(m_radiusValue);
	controlLayout->addWidget(m_numSampleValue);
	controlLayout->addWidget(m_strengthValue);
	controlLayout->addWidget(m_floodAreaCheck);
	controlLayout->addStretch();
	setLayout(controlLayout);
	setTitle(tr("Create"));
	
	connect(m_radiusValue, SIGNAL(valueChanged(double)), this, SLOT(sendRadius(double)));
	connect(m_numSampleValue, SIGNAL(valueChanged(int)), this, SLOT(sendNumSample(int)));
	connect(m_strengthValue, SIGNAL(valueChanged(double)), this, SLOT(sendStrength(double)));
	connect(m_floodAreaCheck, SIGNAL(stateChanged(int)), this, SLOT(sendFloodRegion(int)));
}

double FloodBox::radius() const
{
	return m_radiusValue->value();
}

double FloodBox::strength() const
{
	return m_strengthValue->value();
}

int FloodBox::floodRegion() const
{
	return m_floodAreaCheck->checkState();
}

void FloodBox::sendRadius(double x)
{
	emit radiusChanged(x);
}

void FloodBox::sendStrength(double x)
{
	emit strengthChanged(x);
}

void FloodBox::sendNumSample(int x)
{
	emit numSampleChanged(x);
}

void FloodBox::sendFloodRegion(int x)
{
	emit floodRegionChanged(x);
}
