/*
 *  Phong.cpp
 *  
 *
 *  Created by jian zhang on 10/2/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "Phong.h"
#include "phong_implement.h"
Phong::Phong() : _exposure(100.f), _divideByNdoL(0) 
{
	controlsGroup = new QGroupBox(tr("Phong"));
	
	exposureName = new QLabel(tr("Exposure"));
	exposureValue = new QLineEdit;
	exposureValue->setReadOnly(true);
	exposureValue->setText(tr("100"));
	exposureSlider = new QSlider(Qt::Horizontal);
	exposureSlider->setRange(100, 1000);
	exposureSlider->setSingleStep(1);
	exposureSlider->setValue(100);
	
	divideByNdoLName = new QLabel(tr("Divide By NdotL"));
	divideByNdoLControl = new QCheckBox;
	
	QGridLayout *controlLayout = new QGridLayout;
	controlLayout->setColumnStretch(2, 1);
	controlLayout->addWidget(exposureName, 0, 0);
	controlLayout->addWidget(exposureValue, 0, 1);
    controlLayout->addWidget(exposureSlider, 0, 2);
	controlLayout->addWidget(divideByNdoLName, 1, 0);
	controlLayout->addWidget(divideByNdoLControl, 1, 1);
    controlsGroup->setLayout(controlLayout);
	
	QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(controlsGroup);
	setLayout(layout);
	
	connect(exposureSlider, SIGNAL(valueChanged(int)),
            this, SLOT(setExposureValue(int)));
			
	connect(divideByNdoLControl, SIGNAL(stateChanged(int)),
            this, SLOT(setDivideByNdoLValue(int)));
			
	divideByNdoLControl->setCheckState(Qt::Unchecked);
}

void Phong::setExposureValue(int value)
{
	_exposure = (float)value;
	QString t;
	t.setNum(_exposure);
	exposureValue->setText(t);
}

void Phong::setDivideByNdoLValue(int value)
{
	_divideByNdoL = value;
}

void Phong::run(CUDABuffer * buffer, HemisphereMesh * mesh)
{
	float3 *dptr;
	map(buffer, (void **)&dptr);
	
	float3 fV = {V.x, V.y, V.z};
	float3 fN = {N.x, N.y, N.z};
	
	phong_brdf(dptr, mesh->getGridPhi(), mesh->getGridTheta(), fV, fN, _exposure, _divideByNdoL);

	unmap(buffer);
}