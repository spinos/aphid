/*
 *  PhysicsControl.cpp
 *  cudafem
 *
 *  Created by jian zhang on 9/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include <QIntEditSlider.h>
#include <QDoubleEditSlider.h>
#include <QSplineEdit.h>
#include <QPolarCoordinateEdit.h>
#include "PhysicsControl.h"

PhysicsControl::PhysicsControl(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("Physics Control"));
    
    m_youngModulusValue = new QDoubleEditSlider(tr("Young's modulus"), this);
	m_youngModulusValue->setLimit(40000.0, 800000.0);
	m_youngModulusValue->setValue(160000.0);
	
	YGrp = new QGroupBox;
    QHBoxLayout * yLayout = new QHBoxLayout;
	yLayout->addWidget(m_youngModulusValue);
	yLayout->setStretch(1, 1);
	
    YGrp->setLayout(yLayout);
    
    stiffnessCurveLabel = new QLabel(tr("Stiffness curve"));
    m_youngAttenuateValue = new QSplineEdit(this);
    
    yAGrp = new QGroupBox;
    QVBoxLayout * yaLayout = new QVBoxLayout;
    yaLayout->addWidget(stiffnessCurveLabel);
	yaLayout->addWidget(m_youngAttenuateValue);
	yaLayout->setStretch(1, 1);
	
    yAGrp->setLayout(yaLayout);
	
	m_densityValue = new QDoubleEditSlider(tr("Density"), this);
	m_densityValue->setLimit(0.1, 200.0);
	m_densityValue->setValue(100.0);
	
	dsGrp = new QGroupBox;
    QVBoxLayout * dsLayout = new QVBoxLayout;
    dsLayout->addWidget(m_densityValue);
	dsLayout->setStretch(1, 1);
	
    dsGrp->setLayout(dsLayout);
	
	m_windSpeedValue = new QDoubleEditSlider(tr("Wind speed"), this);
	m_windSpeedValue->setLimit(0.0, 30.0);
    m_windSpeedValue->setValue(0.0);
    m_windVecValue = new QPolarCoordinateEdit(tr("Wind vector"), this);
	QGroupBox * windGrp = new QGroupBox;
	QVBoxLayout * windLayout = new QVBoxLayout;
	windLayout->addWidget(m_windVecValue);
	windLayout->addWidget(m_windSpeedValue);
	windLayout->setStretch(1, 1);
	windLayout->setSpacing(2);
	windGrp->setLayout(windLayout);
	
    QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(dsGrp);
	layout->addWidget(YGrp);
	layout->addWidget(yAGrp);
	layout->addWidget(windGrp);
	layout->setStretch(3, 1);
	layout->setSpacing(4);
	
	setLayout(layout);
    
    connect(m_densityValue, SIGNAL(valueChanged(double)), this, SLOT(sendDensity(double)));
	connect(m_youngModulusValue, SIGNAL(valueChanged(double)), this, SLOT(sendYoungModulus(double)));
    connect(m_youngAttenuateValue, SIGNAL(valueChanged(QPointF)), this, SLOT(sendStiffnessAttenuateEnds(QPointF)));
    connect(m_youngAttenuateValue, SIGNAL(leftControlChanged(QPointF)), this, SLOT(sendStiffnessAttenuateLeft(QPointF)));
    connect(m_youngAttenuateValue, SIGNAL(rightControlChanged(QPointF)), this, SLOT(sendStiffnessAttenuateRight(QPointF)));
	connect(m_windSpeedValue, SIGNAL(valueChanged(double)), this, SLOT(sendWindSpeed(double)));
	connect(m_windVecValue, SIGNAL(valueChanged(QPointF)), this, SLOT(sendWindVec(QPointF)));
}

void PhysicsControl::sendDensity(double x)
{ emit densityChanged(x); }

void PhysicsControl::sendYoungModulus(double x)
{ emit youngsModulusChanged(x); }

void PhysicsControl::sendStiffnessAttenuateEnds(QPointF v)
{ emit stiffnessAttenuateEndsChanged(v); }

void PhysicsControl::sendStiffnessAttenuateLeft(QPointF v)
{ emit stiffnessAttenuateLeftChanged(v); }

void PhysicsControl::sendStiffnessAttenuateRight(QPointF v)
{ emit stiffnessAttenuateRightChanged(v); }

void PhysicsControl::sendWindSpeed(double x)
{ emit windSpeedChanged(x); }

void PhysicsControl::sendWindVec(QPointF v)
{ emit windVecChanged(v); }
//:~