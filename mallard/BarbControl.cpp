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
#include <MlFeather.h>
#include <MlFeatherCollection.h>
BarbControl::BarbControl(QWidget *parent)
    : QWidget(parent)
{
	m_waiting = 1;
	m_seedValue = new QIntEditSlider(tr("Random Seed"));
	m_seedValue->setLimit(0, 10000);
	m_seedValue->setValue(99);
	
	m_gridShaftValue = new QIntEditSlider(tr("Grid Along Shaft"));
	m_gridShaftValue->setLimit(2, 100);
	m_gridShaftValue->setValue(10);
	
	m_gridBarbValue = new QIntEditSlider(tr("Grid Along Barb"));
	m_gridBarbValue->setLimit(3, 100);
	m_gridBarbValue->setValue(9);
	
	m_separateCountValue = new QIntEditSlider(tr("Num Separate"));
	m_separateCountValue->setLimit(2, 20);
	m_separateCountValue->setValue(2);
	
	m_separateWeightValue = new QDoubleEditSlider(tr("Separate Strength"));
	m_separateWeightValue->setLimit(0.0, 1.0);
	m_separateWeightValue->setValue(0.0);
	
	m_fuzzyValue = new QDoubleEditSlider(tr("Fuzziness"));
	m_fuzzyValue->setLimit(0.0, 1.0);
	m_fuzzyValue->setValue(0.0);
	
	m_lodValue = new QDoubleEditSlider(tr("Level of Detail"));
	m_lodValue->setLimit(0.05, 1.0);
	m_lodValue->setValue(1.0);
	
	m_barbShrinkValue = new QDoubleEditSlider(tr("Barb Shrink"));
	m_barbShrinkValue->setLimit(0.0, .99);
	m_barbShrinkValue->setValue(.5);
	
	m_shaftShrinkValue = new QDoubleEditSlider(tr("Shaft Shrink"));
	m_shaftShrinkValue->setLimit(0.0, .99);
	m_shaftShrinkValue->setValue(.5);
	
	m_barbWidthScaleValue = new QDoubleEditSlider(tr("Barb Width Scale"));
	m_barbWidthScaleValue->setLimit(0.01, .99);
	m_barbWidthScaleValue->setValue(.67);
	
	QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(m_gridShaftValue);
	layout->addWidget(m_gridBarbValue);
	layout->addWidget(m_separateCountValue);
	layout->addWidget(m_separateWeightValue);
	layout->addWidget(m_fuzzyValue);
	layout->addWidget(m_barbWidthScaleValue);
	layout->addWidget(m_barbShrinkValue);
	layout->addWidget(m_shaftShrinkValue);
	layout->addWidget(m_seedValue);
	layout->addWidget(m_lodValue);
	layout->addStretch(0);
	setLayout(layout);
	
	connect(m_seedValue, SIGNAL(valueChanged(int)), this, SLOT(sendSeed(int)));
	connect(m_separateCountValue, SIGNAL(valueChanged(int)), this, SLOT(sendNumSeparate(int)));
	connect(m_separateWeightValue, SIGNAL(valueChanged(double)), this, SLOT(sendSeparateStrength(double)));
	connect(m_fuzzyValue, SIGNAL(valueChanged(double)), this, SLOT(sendFuzzy(double)));
	connect(m_gridShaftValue, SIGNAL(valueChanged(int)), this, SLOT(sendGridShaft(int)));
	connect(m_gridBarbValue, SIGNAL(valueChanged(int)), this, SLOT(sendGridBarb(int)));
	connect(m_lodValue, SIGNAL(valueChanged(double)), this, SLOT(sendLod(double)));
	connect(m_barbShrinkValue, SIGNAL(valueChanged(double)), this, SLOT(sendBarbShrink(double)));
	connect(m_shaftShrinkValue, SIGNAL(valueChanged(double)), this, SLOT(sendShaftShrink(double)));
	connect(m_barbWidthScaleValue, SIGNAL(valueChanged(double)), this, SLOT(sendWidthScale(double)));
}

void BarbControl::sendSeed(int s)
{
	emit seedChanged(s);
	if(!m_waiting) emit shapeChanged();
}

void BarbControl::sendNumSeparate(int n)
{
	emit numSeparateChanged(n);
	if(!m_waiting) emit shapeChanged();
}

void BarbControl::sendSeparateStrength(double k)
{
	emit separateStrengthChanged(k);
	if(!m_waiting) emit shapeChanged();
}

void BarbControl::sendFuzzy(double fuz)
{
	emit fuzzinessChanged(fuz);
	if(!m_waiting) emit shapeChanged();
}

void BarbControl::sendGridShaft(int g)
{
	emit resShaftChanged(g);
	if(!m_waiting) emit shapeChanged();
}

void BarbControl::sendGridBarb(int g)
{
	emit resBarbChanged(g);
	if(!m_waiting) emit shapeChanged();
}

void BarbControl::sendLod(double l)
{
	emit lodChanged(l);
	if(!m_waiting) emit shapeChanged();
}

void BarbControl::sendBarbShrink(double x)
{
	emit barbShrinkChanged(x);
	if(!m_waiting) emit shapeChanged();
}

void BarbControl::sendShaftShrink(double x)
{
	emit shaftShrinkChanged(x);
	if(!m_waiting) emit shapeChanged();
}

void BarbControl::sendWidthScale(double x)
{
	emit widthScaleChanged(x);
	if(!m_waiting) emit shapeChanged();
}

void BarbControl::receiveSelectionChanged()
{
	MlFeather *f = selectedExample();
	if(!f) return;
	
	m_waiting = 1;
	m_gridShaftValue->setValue(f->resShaft());
	m_gridBarbValue->setValue(f->resBarb());
	m_separateCountValue->setValue(f->numSeparate());
	m_separateWeightValue->setValue(f->separateStrength());
	m_fuzzyValue->setValue(f->fuzzy());
	m_barbShrinkValue->setValue(f->m_barbShrink);
	m_shaftShrinkValue->setValue(f->m_shaftShrink);
	m_barbWidthScaleValue->setValue(f->m_barbWidthScale);
	m_waiting = 0;
	emit exampleChanged();
}
