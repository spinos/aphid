/*
 *  BrushControl.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BrushControl.h"

#include <QtGui>

#include "BrushControl.h"

BrushControl::BrushControl(QWidget *parent)
    : QDialog(parent)
{
	controlsGroup = new QGroupBox(tr("Brush"));
	radiusValue = new QLineEdit;
	radiusValue->setReadOnly(true);
	radius = new QLabel(tr("Radius"));
	radiusSlider = new QSlider(Qt::Horizontal);
	radiusSlider->setRange(1, 100);
	radiusSlider->setSingleStep(1);
	
	pitchValue = new QLineEdit;
	pitchValue->setReadOnly(true);
	pitch = new QLabel(tr("Pitch Angle"));
	pitchSlider = new QSlider(Qt::Horizontal);
	pitchSlider->setRange(1, 100);
	pitchSlider->setSingleStep(1);
	
	QGridLayout *controlLayout = new QGridLayout;
	controlLayout->setColumnStretch(2, 1);
	controlLayout->addWidget(radius, 0, 0);
	controlLayout->addWidget(radiusValue, 0, 1);
    controlLayout->addWidget(radiusSlider, 0, 2);
    controlLayout->addWidget(pitch, 1, 0);
	controlLayout->addWidget(pitchValue, 1, 1);
    controlLayout->addWidget(pitchSlider, 1, 2);
	controlsGroup->setLayout(controlLayout);
	
	QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(controlsGroup);
	setLayout(layout);
	
	connect(radiusSlider, SIGNAL(valueChanged(int)),
            this, SLOT(radiusSliderChanged(int)));
			
	connect(pitchSlider, SIGNAL(valueChanged(int)),
            this, SLOT(pitchSliderChanged(int)));
			
	radiusSlider->setValue(50);
	pitchSlider->setValue(50);

    layout->setSizeConstraint(QLayout::SetMinimumSize);

    setWindowTitle(tr("Brush Settings"));
}

void BrushControl::radiusSliderChanged(int value)
{
    double r = (double)value / 100.f;
	QString t;
	t.setNum(r);
	radiusValue->setText(t);
}

void BrushControl::pitchSliderChanged(int value)
{
    double r = (double)value / 100.f;
	QString t;
	t.setNum(r);
	pitchValue->setText(t);
}
