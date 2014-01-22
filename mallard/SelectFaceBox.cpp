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
	QVBoxLayout * controlLayout = new QVBoxLayout;
	controlLayout->addWidget(img);
	controlLayout->addWidget(m_radiusValue);
	controlLayout->addStretch();
	setLayout(controlLayout);
	setTitle(tr("Select Face"));
	
	connect(m_radiusValue, SIGNAL(valueChanged(double)), this, SLOT(sendRadius(double)));
}

double SelectFaceBox::radius() const
{
	return m_radiusValue->value();
}

void SelectFaceBox::sendRadius(double x)
{
	emit radiusChanged(x);
}