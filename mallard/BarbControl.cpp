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
	
	m_gridShaftValue = new QIntEditSlider(tr("Grid Along Shaft"));
	m_gridShaftValue->setLimit(10, 1000);
	m_gridShaftValue->setValue(100);
	
	m_gridBarbValue = new QIntEditSlider(tr("Grid Along Barb"));
	m_gridBarbValue->setLimit(2, 1000);
	m_gridBarbValue->setValue(10);
	
	m_separateCountValue = new QIntEditSlider(tr("Num Separate"));
	m_separateCountValue->setLimit(3, 64);
	m_separateCountValue->setValue(11);
	
	m_separateWeightValue = new QDoubleEditSlider(tr("Separate Strength"));
	m_separateWeightValue->setLimit(0.0, 1.0);
	m_separateWeightValue->setValue(0.0);
	
	m_fuzzyValue = new QDoubleEditSlider(tr("Fuzziness"));
	m_fuzzyValue->setLimit(0.0, 1.0);
	m_fuzzyValue->setValue(0.0);
	
	QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(m_gridShaftValue);
	layout->addWidget(m_gridBarbValue);
	layout->addWidget(m_separateCountValue);
	layout->addWidget(m_separateWeightValue);
	layout->addWidget(m_fuzzyValue);
	layout->addWidget(m_seedValue);
	layout->addStretch(0);
	setLayout(layout);
	
	connect(m_seedValue, SIGNAL(valueChanged(int)), this, SLOT(sendSeed(int)));
	connect(m_separateCountValue, SIGNAL(valueChanged(int)), this, SLOT(sendNumSeparate(int)));
	connect(m_separateWeightValue, SIGNAL(valueChanged(double)), this, SLOT(sendSeparateStrength(double)));
	connect(m_fuzzyValue, SIGNAL(valueChanged(double)), this, SLOT(sendFuzzy(double)));
	connect(m_gridShaftValue, SIGNAL(valueChanged(int)), this, SLOT(sendGridShaft(int)));
	connect(m_gridBarbValue, SIGNAL(valueChanged(int)), this, SLOT(sendGridBarb(int)));
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

void BarbControl::sendFuzzy(double f)
{
	emit fuzzyChanged(f);
}

void BarbControl::sendGridShaft(int g)
{
	emit gridShaftChanged(g);
}

void BarbControl::sendGridBarb(int g)
{
	emit gridBarbChanged(g);
}
