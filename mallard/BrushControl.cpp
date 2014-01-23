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

BrushControl::BrushControl(QWidget *parent)
    : QDialog(parent)
{
	selectFace = new SelectFaceBox(this);
	comb = new CombBox(this);
	curl = new CurlBox(this);
	brushScale = new ScaleBox(this);
	flood = new FloodBox(this);
	eraseControl = new EraseBox(this);

	stackLayout = new QStackedLayout(this);
	
	stackLayout->addWidget(flood);
	stackLayout->addWidget(comb);
	stackLayout->addWidget(brushScale);
	stackLayout->addWidget(curl);
	stackLayout->addWidget(eraseControl);
	stackLayout->addWidget(selectFace);
	
	stackLayout->setCurrentIndex(5);
	setLayout(stackLayout);

    stackLayout->setSizeConstraint(QLayout::SetMinimumSize);
	setContentsMargins(8, 8, 8, 8);
	stackLayout->setContentsMargins(0, 0, 0, 0);

    setWindowTitle(tr("Brush Control"));
	
	connect(selectFace, SIGNAL(radiusChanged(double)), this, SLOT(sendBrushRadius(double)));
	connect(selectFace, SIGNAL(twoSidedChanged(int)), this, SLOT(sendBrushTwoSided(int)));
	connect(flood, SIGNAL(radiusChanged(double)), this, SLOT(sendBrushRadius(double)));
	connect(eraseControl, SIGNAL(radiusChanged(double)), this, SLOT(sendBrushRadius(double)));
	connect(comb, SIGNAL(radiusChanged(double)), this, SLOT(sendBrushRadius(double)));
	connect(brushScale, SIGNAL(radiusChanged(double)), this, SLOT(sendBrushRadius(double)));
	connect(curl, SIGNAL(radiusChanged(double)), this, SLOT(sendBrushRadius(double)));
	connect(flood, SIGNAL(strengthChanged(double)), this, SLOT(sendBrushStrength(double)));
	connect(eraseControl, SIGNAL(strengthChanged(double)), this, SLOT(sendBrushStrength(double)));
}

QWidget * BrushControl::floodControlWidget()
{
	return flood;
}

QWidget * BrushControl::eraseControlWidget()
{
    return eraseControl;
}

void BrushControl::receiveToolContext(int c)
{
	double r = flood->radius();
	double s = flood->strength();
	int ts = 0;
	switch(c) {
		case ToolContext::CreateBodyContourFeather:
			stackLayout->setCurrentIndex(0);
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
			s = eraseControl->strength();
			break;
		case ToolContext::SelectFace:
			stackLayout->setCurrentIndex(5);
			r = selectFace->radius();
			ts = selectFace->twoSided();
			break;
		default:
			break;
	}
	sendBrushRadius(r);
	sendBrushStrength(s);
	sendBrushTwoSided(ts);
}
	
void BrushControl::sendBrushRadius(double d)
{
	emit brushRadiusChanged(d);
}
	
void BrushControl::sendBrushStrength(double d)
{
	emit brushStrengthChanged(d);
}

void BrushControl::sendBrushTwoSided(int x)
{
	emit brushTwoSidedChanged(x);
}