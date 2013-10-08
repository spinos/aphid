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
	controlsGroup = new QGroupBox(tr("Frame"));
	QScrollBar * m_bar = new QScrollBar(Qt::Horizontal);
	QLabel * m_minLabel = new QLabel(tr("Minimum"));
	QLabel * m_maxLabel = new QLabel(tr("Maximum"));
	QLabel * m_currentLabel = new QLabel(tr("Current"));
	QSpinBox * m_minSpin = new QSpinBox;
	QSpinBox * m_maxSpin = new QSpinBox;
	QSpinBox * m_currentSpin = new QSpinBox;
	m_bar->setFocusPolicy(Qt::StrongFocus);
	m_bar->setMinimumWidth(400);
	
	QGridLayout *spinLayout = new QGridLayout;
    spinLayout->addWidget(m_minLabel, 0, 0);
    spinLayout->addWidget(m_maxLabel, 1, 0);
    spinLayout->addWidget(m_currentLabel, 2, 0);
    spinLayout->addWidget(m_minSpin, 0, 1);
    spinLayout->addWidget(m_maxSpin, 1, 1);
    spinLayout->addWidget(m_currentSpin, 2, 1);
    
	QHBoxLayout * controlLayout = new QHBoxLayout;
	controlLayout->addLayout(spinLayout);
	controlsGroup->setLayout(controlLayout);
	
	QVBoxLayout *layout = new QVBoxLayout;
	
	layout->addWidget(controlsGroup);
	layout->addWidget(m_bar);
	
	setLayout(layout);

    layout->setSizeConstraint(QLayout::SetMinimumSize);
	setContentsMargins(8, 8, 8, 8);
	layout->setContentsMargins(0, 0, 0, 0);

    setWindowTitle(tr("Time Control"));
    
	
}

