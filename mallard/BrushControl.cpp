/*
 *  BrushControl.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include <QIntEditSlider.h>
#include <QDoubleEditSlider.h>
#include "BrushControl.h"

BrushControl::BrushControl(QWidget *parent)
    : QDialog(parent)
{
	controlsGroup = new QGroupBox(tr("Brush"));
	
	m_numSampleValue = new QIntEditSlider(tr("Num Samples"));
	m_numSampleValue->setLimit(8, 10000);
	
	m_radiusValue = new QDoubleEditSlider(tr("Radius"));
	m_radiusValue->setLimit(0.1, 1000.0);
	
	m_pitchValue = new QDoubleEditSlider(tr("Pitch Angle"));
	m_pitchValue->setLimit(0.1, 1.1);
	
	QVBoxLayout * controlLayout = new QVBoxLayout;
	controlLayout->addWidget(m_radiusValue);
	controlLayout->addWidget(m_pitchValue);
	controlLayout->addWidget(m_numSampleValue);
	
	controlsGroup->setLayout(controlLayout);
	
	QVBoxLayout *layout = new QVBoxLayout;
	
	layout->addWidget(controlsGroup);
	setLayout(layout);

    layout->setSizeConstraint(QLayout::SetMinimumSize);
	setContentsMargins(8, 8, 8, 8);
	layout->setContentsMargins(0, 0, 0, 0);

    setWindowTitle(tr("Brush Settings"));
    
	m_numSampleValue->setValue(32);
	m_radiusValue->setValue(4.0);
	m_pitchValue->setValue(0.6);
}

QWidget * BrushControl::numSamplesWidget()
{
	return m_numSampleValue;
}

QWidget * BrushControl::radiusWidget()
{
	return m_radiusValue;
}

QWidget * BrushControl::pitchWidget()
{
	return m_pitchValue;
}
