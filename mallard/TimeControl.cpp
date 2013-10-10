/*
 *  TimeControl.cpp
 *  mallard
 *
 *  Created by jian zhang on 10/8/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
 
#include <QtGui>
#include <QIntEditSlider.h>
#include <QDoubleEditSlider.h>
#include "TimeControl.h"

TimeControl::TimeControl(QWidget *parent)
    : QDialog(parent)
{
	minGroup = new QGroupBox;
	minLabel = new QLabel;
	minSpin = new QSpinBox;
	QHBoxLayout *minLayout = new QHBoxLayout;
	minLayout->setContentsMargins(4, 4, 4, 4);
	minLayout->addWidget(minLabel);
	minLayout->addWidget(minSpin);
	minGroup->setLayout(minLayout);
	
	maxGroup = new QGroupBox;
	maxLabel = new QLabel;
	maxSpin = new QSpinBox;
	QHBoxLayout *maxLayout = new QHBoxLayout;
	maxLayout->setContentsMargins(4, 4, 4, 4);
	maxLayout->addWidget(maxSpin);
	maxLayout->addWidget(maxLabel);
	maxGroup->setLayout(maxLayout);
	
	
	playGroup = new QGroupBox;
	bar = new QScrollBar(Qt::Horizontal);
	currentSpin = new QSpinBox;
	bar->setFocusPolicy(Qt::StrongFocus);
	bar->setMinimumWidth(400);
	QHBoxLayout *playLayout = new QHBoxLayout;
	playLayout->setContentsMargins(4, 4, 4, 4);
	playLayout->addWidget(bar);
	playLayout->addWidget(currentSpin);
	playLayout->setStretch(0, 1);
	playGroup->setLayout(playLayout);

	QHBoxLayout *layout = new QHBoxLayout;
	
	layout->addWidget(minGroup);
	layout->addWidget(playGroup);
	layout->addWidget(maxGroup);
	layout->setStretch(1, 1);
	layout->setSpacing(4);
	
	setLayout(layout);

    layout->setSizeConstraint(QLayout::SetMinimumSize);
	setContentsMargins(4, 4, 4, 4);
	layout->setContentsMargins(0, 0, 0, 0);

    setWindowTitle(tr("Time Control"));
	
	setFrameRange(1, 99);
	
	disableControl();
    
	connect(bar, SIGNAL(valueChanged(int)), this, SLOT(updateCurrentFrame(int)));
	connect(currentSpin, SIGNAL(valueChanged(int)), this, SLOT(updateCurrentFrame(int)));
	connect(minSpin, SIGNAL(editingFinished()), this, SLOT(setPlayMin()));
	connect(maxSpin, SIGNAL(editingFinished()), this, SLOT(setPlayMax()));
}

void TimeControl::setPlayMin()
{
	int x = minSpin->value();
	if(x >= maxSpin->value()) {
		maxSpin->setValue(x + 1);
		bar->setMaximum(x + 1);
		currentSpin->setMaximum(x + 1);
	}
	
	if(currentFrame() < x)
		updateCurrentFrame(x);

	bar->setMinimum(x);
	currentSpin->setMinimum(x);
}

void TimeControl::setPlayMax()
{
	int x = maxSpin->value();
	if(x <= minSpin->value()) {
		minSpin->setValue(x - 1);
		bar->setMinimum(x - 1);
		currentSpin->setMinimum(x - 1);
	}
	
	if(currentFrame() > x)
		updateCurrentFrame(x);
		
	bar->setMaximum(x);
	currentSpin->setMaximum(x);
}

void TimeControl::setFrameRange(int mn, int mx)
{
	minLabel->setText(QString("%1").arg(mn));
	minSpin->setMinimum(mn);
	minSpin->setMaximum(mx - 1);
	minSpin->setValue(mn);
	
	maxLabel->setText(QString("%1").arg(mx));
	maxSpin->setMinimum(mn + 1);
	maxSpin->setMaximum(mx);
	maxSpin->setValue(mx);
	
	bar->setMinimum(mn);
	bar->setMaximum(mx);
	bar->setValue(mn);
	currentSpin->setMinimum(mn);
	currentSpin->setMaximum(mx);
	currentSpin->setValue(mn);
	PlaybackControl::setFrameRange(mn, mx);
}

void TimeControl::updateCurrentFrame(int x)
{
	if(bar->value() != x)
		bar->setValue(x);
	if(currentSpin->value() != x)
		currentSpin->setValue(x);
		
	if(currentFrame() != x) {
		setCurrentFrame(x);
		emit currentFrameChanged(x);
	}
}

void TimeControl::disableControl()
{
	minGroup->setEnabled(false);
	playGroup->setEnabled(false);
	maxGroup->setEnabled(false);
	PlaybackControl::disableControl();
}

void TimeControl::enableControl()
{
	minGroup->setEnabled(true);
	playGroup->setEnabled(true);
	maxGroup->setEnabled(true);
	PlaybackControl::enableControl();
}
