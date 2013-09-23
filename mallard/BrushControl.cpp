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
	
	maxR = new QLabel(tr("Max Radius"));
	maxRVal = new QDoubleValidator;
	maxRVal->setBottom(0.1);
	maxRVal->setRange(0.1, 1000.0, 4);
	maxRValue = new QLineEdit;
	maxRValue->setValidator(maxRVal);
	maxRValue->setText(tr("4.0"));
	
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
	
	numSamples = new QLabel(tr("Num Feathers"));
	numSamplesValue = new QLineEdit;
	numSamplesVal = new QIntValidator;
	numSamplesVal->setBottom(8);
	numSamplesVal->setRange(8, 10000);
	numSamplesValue->setValidator(numSamplesVal);
	
	QGridLayout *controlLayout = new QGridLayout;
	controlLayout->setColumnStretch(2, 1);
	controlLayout->addWidget(maxR, 0, 0);
	controlLayout->addWidget(maxRValue, 0, 1);
	controlLayout->addWidget(radius, 1, 0);
	controlLayout->addWidget(radiusValue, 1, 1);
    controlLayout->addWidget(radiusSlider, 1, 2);
    controlLayout->addWidget(numSamples, 2, 0);
	controlLayout->addWidget(numSamplesValue, 2, 1);
	controlLayout->addWidget(pitch, 3, 0);
	controlLayout->addWidget(pitchValue, 3, 1);
    controlLayout->addWidget(pitchSlider, 3, 2);
	controlsGroup->setLayout(controlLayout);
	
	QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(controlsGroup);
	setLayout(layout);
	
	connect(maxRValue, SIGNAL(editingFinished()),
            this, SLOT(maxRadiusEdited()));
	
	connect(radiusSlider, SIGNAL(valueChanged(int)),
            this, SLOT(radiusSliderChanged(int)));
			
	connect(pitchSlider, SIGNAL(valueChanged(int)),
            this, SLOT(pitchSliderChanged(int)));
    
    connect(numSamplesValue, SIGNAL(editingFinished()),
            this, SLOT(numSamplesEdited()));
			
	radiusSlider->setValue(50);
	pitchSlider->setValue(50);
	numSamplesValue->setText(tr("32"));

    layout->setSizeConstraint(QLayout::SetMinimumSize);

    setWindowTitle(tr("Brush Settings"));
    
    emit radiusChanged(2.0);
    emit numSamplesChanged(32);
    emit pitchChanged(0.5);
}

void BrushControl::radiusSliderChanged(int value)
{
    double r = (double)value / 100.0 * maxRValue->text().toDouble();
	QString t;
	t.setNum(r);
	radiusValue->setText(t);
	emit radiusChanged(r);
}

void BrushControl::pitchSliderChanged(int value)
{
    double r = (double)value / 100.0 * 0.9 + 0.1;
	QString t;
	t.setNum(r);
	pitchValue->setText(t);
	emit pitchChanged(r);
}

void BrushControl::maxRadiusEdited()
{
    double r = maxRValue->text().toDouble();
    
    if(r < maxRVal->bottom()) r = maxRVal->bottom();
    else if(r > maxRVal->top()) r = maxRVal->top();
    
    QString t;
	t.setNum(r);
	maxRValue->setText(t);
	
    r = r * (double)radiusSlider->value()/ 100.0 ;
    
    t.setNum(r);
	radiusValue->setText(t);
	
	emit radiusChanged(r);
}

void BrushControl::numSamplesEdited()
{
    int s = numSamplesValue->text().toInt();
    if(s < numSamplesVal->bottom()) s = numSamplesVal->bottom();
    else if(s > numSamplesVal->top()) s = numSamplesVal->top();
    
    QString t;
	t.setNum(s);
	numSamplesValue->setText(t);
    emit numSamplesChanged(s);
}
