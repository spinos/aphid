/*
 *  SelectFaceBox.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/23/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "SelectFaceBox.h"
#include <QtGui>
#include <QDoubleEditSlider.h>

SelectFaceBox::SelectFaceBox(QWidget *parent) : QGroupBox(parent)
{
	QPixmap *pix = new QPixmap(":selectFaceActive.png");
	QLabel * img = new QLabel;
	img->setPixmap(*pix);
	m_radiusValue = new QDoubleEditSlider(tr("Radius"));
	m_radiusValue->setLimit(0.01, 1000.0);
	m_radiusValue->setValue(1.0);
	m_twoSidedCheck = new QCheckBox(tr("Front And Back Side"));
	QVBoxLayout * controlLayout = new QVBoxLayout;
	controlLayout->addWidget(img);
	controlLayout->addWidget(m_radiusValue);
	controlLayout->addWidget(m_twoSidedCheck);
	controlLayout->addStretch();
	setLayout(controlLayout);
	setTitle(tr("Select Face"));
	
	connect(m_radiusValue, SIGNAL(valueChanged(double)), this, SLOT(sendRadius(double)));
	connect(m_twoSidedCheck, SIGNAL(stateChanged(int)), this, SLOT(sendTwoSided(int)));
}

double SelectFaceBox::radius() const
{
	return m_radiusValue->value();
}

int SelectFaceBox::twoSided() const
{
	return m_twoSidedCheck->checkState();
}

void SelectFaceBox::sendRadius(double x)
{
	emit radiusChanged(x);
}

void SelectFaceBox::sendTwoSided(int x)
{
	emit twoSidedChanged(x);
}