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
#include "SelectFaceBox.h"
#include "CombBox.h"
#include "CurlBox.h"
#include "ScaleBox.h"
#include "FloodBox.h"
#include "EraseBox.h"
#include "PaintBox.h"
#include <BaseBrush.h>

BrushControl::BrushControl(BaseBrush * brush, QWidget *parent)
    : QDialog(parent)
{
	m_brush = brush;
	selectFace = new SelectFaceBox(this);
	comb = new CombBox(this);
	curl = new CurlBox(this);
	brushScale = new ScaleBox(this);
	flood = new FloodBox(this);
	eraseControl = new EraseBox(this);
	paintControl = new PaintBox(this);

	stackLayout = new QStackedLayout(this);
	
	stackLayout->addWidget(flood);
	stackLayout->addWidget(comb);
	stackLayout->addWidget(brushScale);
	stackLayout->addWidget(curl);
	stackLayout->addWidget(eraseControl);
	stackLayout->addWidget(selectFace);
	stackLayout->addWidget(paintControl);
	
	stackLayout->setCurrentIndex(5);
	setLayout(stackLayout);

    stackLayout->setSizeConstraint(QLayout::SetMinimumSize);
	setContentsMargins(8, 8, 8, 8);
	stackLayout->setContentsMargins(0, 0, 0, 0);

    setWindowTitle(tr("Brush Control"));
	
	connect(selectFace, SIGNAL(radiusChanged(double)), this, SLOT(sendBrushRadius(double)));
	connect(flood, SIGNAL(radiusChanged(double)), this, SLOT(sendBrushRadius(double)));
	connect(eraseControl, SIGNAL(radiusChanged(double)), this, SLOT(sendBrushRadius(double)));
	connect(comb, SIGNAL(radiusChanged(double)), this, SLOT(sendBrushRadius(double)));
	connect(brushScale, SIGNAL(radiusChanged(double)), this, SLOT(sendBrushRadius(double)));
	connect(curl, SIGNAL(radiusChanged(double)), this, SLOT(sendBrushRadius(double)));
	connect(paintControl, SIGNAL(radiusChanged(double)), this, SLOT(sendBrushRadius(double)));
	
	connect(flood, SIGNAL(strengthChanged(double)), this, SLOT(sendBrushStrength(double)));
	connect(eraseControl, SIGNAL(strengthChanged(double)), this, SLOT(sendBrushStrength(double)));
	connect(paintControl, SIGNAL(strengthChanged(double)), this, SLOT(sendBrushStrength(double)));
	
	connect(selectFace, SIGNAL(twoSidedChanged(int)), this, SLOT(sendBrushTwoSided(int)));
	
	connect(flood, SIGNAL(floodRegionChanged(int)), this, SLOT(sendBrushFilterByColor(int)));
	connect(eraseControl, SIGNAL(eraseRegionChanged(int)), this, SLOT(sendBrushFilterByColor(int)));
	
	connect(flood, SIGNAL(initialCurlChanged(double)), this, SLOT(sendBrushPitch(double)));
	connect(flood, SIGNAL(numSampleChanged(int)), this, SLOT(sendBrushNumSamples(int)));
	
	connect(paintControl, SIGNAL(colorChanged(QColor)), this, SLOT(sendBrushColor(QColor)));
	connect(paintControl, SIGNAL(dropoffChanged(double)), this, SLOT(sendBrushDropoff(double)));
}

void BrushControl::receiveToolContext(int c)
{
	double r = 1.f;
	double strength = 1.f;
	int ts = 0;
	int byRegion = 0;
	QColor col = paintControl->color();
	double dropoff = 0.0;
	
	switch(c) {
		case ToolContext::CreateBodyContourFeather:
			stackLayout->setCurrentIndex(0);
			r = flood->radius();
			strength = flood->strength();
			byRegion = flood->floodRegion();
			break;
		case ToolContext::CombBodyContourFeather:
			stackLayout->setCurrentIndex(1);
			r = comb->radius();
			break;
		case ToolContext::ScaleBodyContourFeather:
			stackLayout->setCurrentIndex(2);
			r = brushScale->radius();
			break;
		case ToolContext::PitchBodyContourFeather:
			stackLayout->setCurrentIndex(3);
			r = curl->radius();
			break;
		case ToolContext::EraseBodyContourFeather:
			stackLayout->setCurrentIndex(4);
			r = eraseControl->radius();
			strength = eraseControl->strength();
			byRegion = eraseControl->eraseRegion();
			break;
		case ToolContext::SelectFace:
			stackLayout->setCurrentIndex(5);
			r = selectFace->radius();
			ts = selectFace->twoSided();
			break;
		case ToolContext::PaintMap:
			stackLayout->setCurrentIndex(6);
			r = paintControl->radius();
			dropoff = paintControl->dropoff();
			strength = paintControl->strength();
			break;
		default:
			break;
	}
	m_brush->setRadius(r);
	m_brush->setStrength(strength);
	m_brush->setDropoff(dropoff);

	sendBrushTwoSided(ts);
	sendBrushFilterByColor(byRegion);
}
	
void BrushControl::sendBrushRadius(double d)
{
	m_brush->setRadius(d);
	emit brushChanged();
}
	
void BrushControl::sendBrushStrength(double d)
{
	m_brush->setStrength(d);
	emit brushChanged();
}

void BrushControl::sendBrushTwoSided(int x)
{
	if(x == Qt::Checked)
		m_brush->setTwoSided(true);
	else
		m_brush->setTwoSided(false);
	emit brushChanged();
}

void BrushControl::sendBrushFilterByColor(int x)
{
	if(x == Qt::Checked)
		m_brush->setFilterByColor(true);
	else
		m_brush->setFilterByColor(false);
	emit brushChanged();
}

void BrushControl::sendBrushPitch(double d)
{
	m_brush->setPitch(d);
	emit brushChanged();
}

void BrushControl::sendBrushNumSamples(int x)
{
	m_brush->setNumDarts(x);
	emit brushChanged();
}

void BrushControl::sendBrushColor(QColor c)
{
	Float3 colf(c.redF(), c.greenF(), c.blueF());
	m_brush->setColor(colf);
	emit brushChanged();
}

void BrushControl::sendBrushDropoff(double x)
{
	m_brush->setDropoff(x);
	emit brushChanged();
}

