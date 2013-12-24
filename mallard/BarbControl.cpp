/*
 *  BarbControl.cpp
 *  mallard
 *
 *  Created by jian zhang on 12/25/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "BarbControl.h"
#include <QIntEditSlider.h>
#include <QDoubleEditSlider.h>

BarbControl::BarbControl(QWidget *parent)
    : QWidget(parent)
{
	m_seedValue = new QIntEditSlider(tr("Random Seed"));
	m_seedValue->setLimit(0, 10000);
	m_seedValue->setValue(99);
	
	m_separateCountValue = new QIntEditSlider(tr("Num Separate"));
	m_separateCountValue->setLimit(3, 64);
	m_separateCountValue->setValue(11);
	
	m_separateWeightValue = new QDoubleEditSlider(tr("Separate Strength"));
	m_separateWeightValue->setLimit(0.0, 1.0);
	m_separateWeightValue->setValue(0.0);
	
	QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(m_seedValue);
	layout->addWidget(m_separateCountValue);
	layout->addWidget(m_separateWeightValue);
	layout->addStretch(0);
	setLayout(layout);
	
	connect(m_seedValue, SIGNAL(valueChanged(int)), this, SLOT(sendSeed(int)));
	connect(m_separateCountValue, SIGNAL(valueChanged(int)), this, SLOT(sendNumSeparate(int)));
	connect(m_separateWeightValue, SIGNAL(valueChanged(double)), this, SLOT(sendSeparateStrength(double)));
}

void BarbControl::sendSeed(int s)
{
	emit seedChanged(s);
}

void BarbControl::sendNumSeparate(int n)
{
	emit numSeparateChanged(n);
}

void BarbControl::sendSeparateStrength(double k)
{
	emit separateStrengthChanged(k);
}
