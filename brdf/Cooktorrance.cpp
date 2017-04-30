/*
 *  Cooktorrance.cpp
 *  
 *
 *  Created by jian zhang on 10/2/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "Cooktorrance.h"
#include "cooktorrance_implement.h"
Cooktorrance::Cooktorrance() : _m(.1f), _f0(.1f), _includeF(1), _includeG(1)
{
	controlsGroup = new QGroupBox(tr("Cooktorrance"));
	
	mName = new QLabel(tr("m"));
	mValue = new QLineEdit;
	mValue->setReadOnly(true);
	mValue->setText(tr("0.1"));
	mSlider = new QSlider(Qt::Horizontal);
	mSlider->setRange(1, 500);
	mSlider->setSingleStep(1);
	mSlider->setValue(100);
	
	f0Name = new QLabel(tr("f0"));
	f0Value = new QLineEdit;
	f0Value->setReadOnly(true);
	f0Value->setText(tr("0.1"));
	f0Slider = new QSlider(Qt::Horizontal);
	f0Slider->setRange(0, 1000);
	f0Slider->setSingleStep(1);
	f0Slider->setValue(100);
	
	incfName = new QLabel(tr("include F"));
	incfControl = new QCheckBox;
	incfControl->setCheckState(Qt::Checked);
	
	incgName = new QLabel(tr("include G"));
	incgControl = new QCheckBox;
	incgControl->setCheckState(Qt::Checked);
	
	QGridLayout *controlLayout = new QGridLayout;
	controlLayout->setColumnStretch(2, 1);
	controlLayout->addWidget(mName, 0, 0);
	controlLayout->addWidget(mValue, 0, 1);
    controlLayout->addWidget(mSlider, 0, 2);
    controlLayout->addWidget(f0Name, 1, 0);
	controlLayout->addWidget(f0Value, 1, 1);
    controlLayout->addWidget(f0Slider, 1, 2);
    controlsGroup->setLayout(controlLayout);
	
	QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(controlsGroup);
	setLayout(layout);
	
	connect(mSlider, SIGNAL(valueChanged(int)),
            this, SLOT(setMValue(int)));
    
    connect(f0Slider, SIGNAL(valueChanged(int)),
            this, SLOT(setF0Value(int)));
			
	connect(incfControl, SIGNAL(stateChanged(int)),
            this, SLOT(setIncfValue(int)));
			
	connect(incgControl, SIGNAL(stateChanged(int)),
            this, SLOT(setIncgValue(int)));
	
}

void Cooktorrance::setMValue(int value)
{
	_m = (float)value / 1000.f;
	QString t;
	t.setNum(_m);
	mValue->setText(t);
}

void Cooktorrance::setF0Value(int value)
{
	_f0 = (float)value / 1000.f;
	QString t;
	t.setNum(_f0);
	f0Value->setText(t);
}

void Cooktorrance::setIncfValue(int value)
{
	_includeF = value;
}

void Cooktorrance::setIncgValue(int value)
{
	_includeG = value;
}

void Cooktorrance::run(CUDABuffer * buffer, BaseMesh * mesh)
{
	float3 *dptr;
	map(buffer, (void **)&dptr);
	
	float3 fV = {V.x, V.y, V.z};
	float3 fN = {N.x, N.y, N.z};
	
	unsigned width, height;
	calculateDim(mesh->getNumVertices(), width, height);
	
	cooktorrance_brdf(dptr, mesh->getNumVertices(), width, fV, fN, _m, _f0);

	unmap(buffer);
}