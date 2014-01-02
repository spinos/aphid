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
	
	QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(m_gridShaftValue);
	layout->addWidget(m_gridBarbValue);
	layout->addWidget(m_separateCountValue);
	layout->addWidget(m_separateWeightValue);
	layout->addWidget(m_fuzzyValue);
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
}

void BarbControl::sendSeed(int s)
{
	MlFeather *f = selectedExample();
	if(!f) return;
	f->setSeed(s);
	emit shapeChanged();
}

void BarbControl::sendNumSeparate(int n)
{
	MlFeather *f = selectedExample();
	if(!f) return;
	f->setNumSeparate(n);
	emit shapeChanged();
}

void BarbControl::sendSeparateStrength(double k)
{
	MlFeather *f = selectedExample();
	if(!f) return;
	f->setSeparateStrength(k);
	emit shapeChanged();
}

void BarbControl::sendFuzzy(double fuz)
{
	MlFeather *f = selectedExample();
	if(!f) return;
	f->setFuzzy(fuz);
	emit shapeChanged();
}

void BarbControl::sendGridShaft(int g)
{
	MlFeather *f = selectedExample();
	if(!f) return;
	f->setResShaft(g);
	emit shapeChanged();
}

void BarbControl::sendGridBarb(int g)
{
	MlFeather *f = selectedExample();
	if(!f) return;
	f->setResBarb(g);
	emit shapeChanged();
}

void BarbControl::sendLod(double l)
{
	MlFeather *f = selectedExample();
	if(!f) return;
	f->setLevelOfDetail(l);
	emit shapeChanged();
}

void BarbControl::receiveSelectionChanged()
{
	MlFeather *f = selectedExample();
	if(!f) return;
	
	m_gridShaftValue->setValue(f->resShaft());
	m_gridBarbValue->setValue(f->resBarb());
	m_separateCountValue->setValue(f->numSeparate());
	m_separateWeightValue->setValue(f->separateStrength());
	m_fuzzyValue->setValue(f->fuzzy());
	emit shapeChanged();
}
