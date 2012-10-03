/*
 *  Ward.cpp
 *  
 *
 *  Created by jian zhang on 10/2/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "Ward.h"
#include "ward_implement.h"
Ward::Ward() : _alphaX(.25f), _alphaY(.25f), _anisotropic(0)
{
	controlsGroup = new QGroupBox(tr("Ward"));
	
	axName = new QLabel(tr("Alpha X"));
	axValue = new QLineEdit;
	axValue->setReadOnly(true);
	axValue->setText(tr("0.25"));
	axSlider = new QSlider(Qt::Horizontal);
	axSlider->setRange(1, 100);
	axSlider->setSingleStep(1);
	axSlider->setValue(25);
	
	ayName = new QLabel(tr("Alpha Y"));
	ayValue = new QLineEdit;
	ayValue->setReadOnly(true);
	ayValue->setText(tr("0.25"));
	aySlider = new QSlider(Qt::Horizontal);
	aySlider->setRange(1, 100);
	aySlider->setSingleStep(1);
	aySlider->setValue(25);
	
	anisotropicName = new QLabel(tr("Anisotropic"));
	anisotropicControl = new QCheckBox;
	anisotropicControl->setCheckState(Qt::Unchecked);
	
	QGridLayout *controlLayout = new QGridLayout;
	controlLayout->setColumnStretch(2, 1);
	controlLayout->addWidget(axName, 0, 0);
	controlLayout->addWidget(axValue, 0, 1);
    controlLayout->addWidget(axSlider, 0, 2);
    controlLayout->addWidget(ayName, 1, 0);
	controlLayout->addWidget(ayValue, 1, 1);
    controlLayout->addWidget(aySlider, 1, 2);
	controlLayout->addWidget(anisotropicName, 2, 0);
	controlLayout->addWidget(anisotropicControl, 2, 1);
    controlsGroup->setLayout(controlLayout);
	
	QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(controlsGroup);
	setLayout(layout);
	
	connect(axSlider, SIGNAL(valueChanged(int)),
            this, SLOT(setAlphaXValue(int)));
    
    connect(aySlider, SIGNAL(valueChanged(int)),
            this, SLOT(setAlphaYValue(int)));
			
	connect(anisotropicControl, SIGNAL(stateChanged(int)),
            this, SLOT(setAnisotropicValue(int)));		
	
}

void Ward::setAlphaXValue(int value)
{
	_alphaX = (float)value / 100.f;
	QString t;
	t.setNum(_alphaX);
	axValue->setText(t);
}

void Ward::setAlphaYValue(int value)
{
	_alphaY = (float)value / 100.f;
	QString t;
	t.setNum(_alphaY);
	ayValue->setText(t);
}

void Ward::setAnisotropicValue(int value)
{
	_anisotropic = value;
}

void Ward::run(CUDABuffer * buffer, BaseMesh * mesh)
{
	float3 *dptr;
	map(buffer, (void **)&dptr);
	
	float3 fV = {V.x, V.y, V.z};
	float3 fN = {N.x, N.y, N.z};
	float3 fX = {Tangent.x, Tangent.y, Tangent.z};
	float3 fY = {Binormal.x, Binormal.y, Binormal.z};
	bool bAnis = _anisotropic > 0 ? 1 : 0;
	
	unsigned width, height;
	calculateDim(mesh->getNumVertices(), width, height);
	
	ward_brdf(dptr, mesh->getNumVertices(), width, fV, fN, fX, fY, _alphaX, _alphaY, bAnis);

	unmap(buffer);
}