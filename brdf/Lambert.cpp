/*
 *  Lambert.cpp
 *  
 *
 *  Created by jian zhang on 10/2/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "Lambert.h"

#include "lambert_implement.h"

Lambert::Lambert() : _reflectance(0.5f)
{
	controlsGroup = new QGroupBox(tr("Lambert"));
	reflectanceValue = new QLineEdit;
	reflectanceValue->setReadOnly(true);
	reflectance = new QLabel(tr("Reflectance"));
	slider = new QSlider(Qt::Horizontal);
	slider->setRange(1, 100);
	slider->setSingleStep(1);
	
	QGridLayout *controlLayout = new QGridLayout;
	controlLayout->setColumnStretch(2, 1);
	controlLayout->addWidget(reflectance, 0, 0);
	controlLayout->addWidget(reflectanceValue, 0, 1);
    controlLayout->addWidget(slider, 0, 2);
    controlsGroup->setLayout(controlLayout);
	
	QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(controlsGroup);
	setLayout(layout);
	
	connect(slider, SIGNAL(valueChanged(int)),
            this, SLOT(setReflectanceValue(int)));
			
	slider->setValue(50);

}

void Lambert::setReflectanceValue(int value)
{
	_reflectance = (double)value/ 100.f;
	QString t;
	t.setNum(_reflectance);
	reflectanceValue->setText(t);
}

void Lambert::run(CUDABuffer * buffer, BaseMesh * mesh)
{
	float3 *dptr;
	map(buffer, (void **)&dptr);
	
	unsigned width, height;
	calculateDim(mesh->getNumVertices(), width, height);
	
	lambert_brdf(dptr, mesh->getNumVertices(), width, _reflectance);

	unmap(buffer);
}