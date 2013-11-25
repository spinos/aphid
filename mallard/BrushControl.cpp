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
#include <ToolContext.h>
#include "BrushControl.h"

BrushControl::BrushControl(QWidget *parent)
    : QDialog(parent)
{
	createGroup();
	eraseGroup();
	combGroup();
	scaleGroup();
	bendGroup();
	
	stackLayout = new QStackedLayout(this);
	
	stackLayout->addWidget(controlsGroupC);
	stackLayout->addWidget(controlsGroupB);
	stackLayout->addWidget(controlsGroupS);
	stackLayout->addWidget(controlsGroupD);
	stackLayout->addWidget(controlsGroupE);
	
	stackLayout->setCurrentIndex(0);
	setLayout(stackLayout);

    stackLayout->setSizeConstraint(QLayout::SetMinimumSize);
	setContentsMargins(8, 8, 8, 8);
	stackLayout->setContentsMargins(0, 0, 0, 0);

    setWindowTitle(tr("Brush Control"));
	
	connect(m_radiusValueC, SIGNAL(valueChanged(double)), this, SLOT(sendBrushRadius(double)));
	connect(m_radiusValueE, SIGNAL(valueChanged(double)), this, SLOT(sendBrushRadius(double)));
	connect(m_radiusValueB, SIGNAL(valueChanged(double)), this, SLOT(sendBrushRadius(double)));
	connect(m_radiusValueS, SIGNAL(valueChanged(double)), this, SLOT(sendBrushRadius(double)));
	connect(m_radiusValueD, SIGNAL(valueChanged(double)), this, SLOT(sendBrushRadius(double)));
	connect(m_createStrengthValue, SIGNAL(valueChanged(double)), this, SLOT(sendBrushStrength(double)));
	connect(m_eraseStrengthValue, SIGNAL(valueChanged(double)), this, SLOT(sendBrushStrength(double)));
}

QWidget * BrushControl::numSamplesWidget()
{
	return m_numSampleValueC;
}

QWidget * BrushControl::radiusWidget()
{
	return m_radiusValueC;
}

QWidget * BrushControl::pitchWidget()
{
	return m_pitchValueC;
}

QWidget * BrushControl::floodRegionWidget()
{
	return m_floodAreaCheck;
}

QWidget * BrushControl::eraseRegionWidget()
{
    return m_eraseAreaCheck;
}

void BrushControl::createGroup()
{
	controlsGroupC = new QGroupBox(tr("Create"));
	
	m_numSampleValueC = new QIntEditSlider(tr("Num Samples"));
	m_numSampleValueC->setLimit(8, 10000);
	
	m_radiusValueC = new QDoubleEditSlider(tr("Radius"));
	m_radiusValueC->setLimit(0.1, 1000.0);
	
	m_pitchValueC = new QDoubleEditSlider(tr("Pitch Angle"));
	m_pitchValueC->setLimit(0.1, 1.1);
	
	m_createStrengthValue = new QDoubleEditSlider(tr("Strength"));
	m_createStrengthValue->setLimit(0.01, 1.0);
	m_createStrengthValue->setValue(1.0);
	
	m_floodAreaCheck = new QCheckBox(tr("Flood Selected Region"));
	
	QVBoxLayout * controlLayout = new QVBoxLayout;
	controlLayout->addWidget(m_radiusValueC);
	controlLayout->addWidget(m_pitchValueC);
	controlLayout->addWidget(m_numSampleValueC);
	controlLayout->addWidget(m_createStrengthValue);
	controlLayout->addWidget(m_floodAreaCheck);
	controlLayout->addStretch();
	controlsGroupC->setLayout(controlLayout);
	
	m_numSampleValueC->setValue(99);
	m_radiusValueC->setValue(4.0);
	m_pitchValueC->setValue(0.1);
}

void BrushControl::eraseGroup()
{
	controlsGroupE = new QGroupBox(tr("Erase"));
	
	m_radiusValueE = new QDoubleEditSlider(tr("Radius"));
	m_radiusValueE->setLimit(0.1, 1000.0);
	m_eraseStrengthValue = new QDoubleEditSlider(tr("Strength"));
	m_eraseStrengthValue->setLimit(0.01, 1.0);
	m_eraseStrengthValue->setValue(1.0);
	m_eraseAreaCheck = new QCheckBox(tr("Erase Within Selected Region"));
	
	QVBoxLayout * controlLayout = new QVBoxLayout;
	controlLayout->addWidget(m_radiusValueE);
	controlLayout->addWidget(m_eraseStrengthValue);
	controlLayout->addWidget(m_eraseAreaCheck);
	controlLayout->addStretch();
	controlsGroupE->setLayout(controlLayout);
}

void BrushControl::combGroup()
{
	controlsGroupB = new QGroupBox(tr("Comb"));
	
	m_radiusValueB = new QDoubleEditSlider(tr("Radius"));
	m_radiusValueB->setLimit(0.1, 1000.0);
	
	QVBoxLayout * controlLayout = new QVBoxLayout;
	controlLayout->addWidget(m_radiusValueB);
	controlLayout->addStretch();
	controlsGroupB->setLayout(controlLayout);
}

void BrushControl::scaleGroup()
{
	controlsGroupS = new QGroupBox(tr("Scale"));
	
	m_radiusValueS = new QDoubleEditSlider(tr("Radius"));
	m_radiusValueS->setLimit(0.1, 1000.0);
	
	QVBoxLayout * controlLayout = new QVBoxLayout;
	controlLayout->addWidget(m_radiusValueS);
	controlLayout->addStretch();
	controlsGroupS->setLayout(controlLayout);
}

void BrushControl::bendGroup()
{
	controlsGroupD = new QGroupBox(tr("Pitch"));
	
	m_radiusValueD = new QDoubleEditSlider(tr("Radius"));
	m_radiusValueD->setLimit(0.1, 1000.0);
	
	QVBoxLayout * controlLayout = new QVBoxLayout;
	controlLayout->addWidget(m_radiusValueD);
	controlLayout->addStretch();
	controlsGroupD->setLayout(controlLayout);
}

void BrushControl::receiveToolContext(int c)
{
	double r = m_radiusValueC->value();
	double s = m_createStrengthValue->value();
	switch(c) {
		case ToolContext::CreateBodyContourFeather:
			stackLayout->setCurrentIndex(0);
			break;
		case ToolContext::CombBodyContourFeather:
			stackLayout->setCurrentIndex(1);
			r = m_radiusValueB->value();
			break;
		case ToolContext::ScaleBodyContourFeather:
			stackLayout->setCurrentIndex(2);
			r = m_radiusValueS->value();
			break;
		case ToolContext::PitchBodyContourFeather:
			stackLayout->setCurrentIndex(3);
			r = m_radiusValueB->value();
			break;
		case ToolContext::EraseBodyContourFeather:
			stackLayout->setCurrentIndex(4);
			r = m_radiusValueE->value();
			s = m_eraseStrengthValue->value();
			break;
		default:
			break;
	}
	sendBrushRadius(r);
	sendBrushStrength(s);
}
	
void BrushControl::sendBrushRadius(double d)
{
	emit brushRadiusChanged(d);
}
	
void BrushControl::sendBrushStrength(double d)
{
	emit brushStrengthChanged(d);
}
