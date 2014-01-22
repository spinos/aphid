/*
 *  EraseBox.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/23/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "EraseBox.h"

#include <QtGui>
#include <QDoubleEditSlider.h>
#include <QIntEditSlider.h>

EraseBox::EraseBox(QWidget *parent) : QGroupBox(parent)
{
	QPixmap *pix = new QPixmap(":eraserActive.png");
	QLabel * img = new QLabel;
	img->setPixmap(*pix);
	m_radiusValue = new QDoubleEditSlider(tr("Radius"));
	m_radiusValue->setLimit(0.01, 1000.0);
	m_radiusValue->setValue(1.0);
	
	m_strengthValue = new QDoubleEditSlider(tr("Strength"));
	m_strengthValue->setLimit(0.01, 1.0);
	m_strengthValue->setValue(1.0);
	
	m_areaCheck = new QCheckBox(tr("Erase Within Selected Region"));
	
	QVBoxLayout * controlLayout = new QVBoxLayout;
	controlLayout->addWidget(img);
	controlLayout->addWidget(m_radiusValue);
	controlLayout->addWidget(m_strengthValue);
	controlLayout->addWidget(m_areaCheck);
	controlLayout->addStretch();
	setLayout(controlLayout);
	setTitle(tr("Erase"));
	
	connect(m_radiusValue, SIGNAL(valueChanged(double)), this, SLOT(sendRadius(double)));
	connect(m_strengthValue, SIGNAL(valueChanged(double)), this, SLOT(sendStrength(double)));
	connect(m_areaCheck, SIGNAL(stateChanged(int)), this, SLOT(sendEraseRegion(int)));
}

double EraseBox::radius() const
{
	return m_radiusValue->value();
}

double EraseBox::strength() const
{
	return m_strengthValue->value();
}

void EraseBox::sendRadius(double x)
{
	emit radiusChanged(x);
}

void EraseBox::sendStrength(double x)
{
	emit strengthChanged(x);
}

void EraseBox::sendEraseRegion(int x)
{
	emit eraseRegionChanged(x);
}