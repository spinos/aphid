/*
 *  GenTetControl.cpp
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
#include "GenTetControl.h"

GenTetControl::GenTetControl(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(tr("Generate Tetrahedron Control"));
    
    m_estimateNValue = new QDoubleEditSlider(tr("Estimate n groups"), this);
	m_estimateNValue->setLimit(100.0, 10000.0);
	m_estimateNValue->setValue(2000.0);
	
	QGroupBox * estimateNGrp = new QGroupBox;
    QHBoxLayout * yLayout = new QHBoxLayout;
	yLayout->addWidget(m_estimateNValue);
	yLayout->setStretch(1, 1);
	
    estimateNGrp->setLayout(yLayout);
    
    m_rebuildAct = new QPushButton(tr("Rebuild tetrahedron mesh"));
	
    QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(estimateNGrp);
    layout->addWidget(m_rebuildAct);
	layout->setStretch(3, 1);
	layout->setSpacing(4);
	
	setLayout(layout);
    
    connect(m_rebuildAct, SIGNAL(clicked()), this, SLOT(sendRebuild()));
}

void GenTetControl::sendRebuild()
{ 
    double n = m_estimateNValue->value();
    emit rebuildTet(n); 
}

void GenTetControl::receiveEstimatedN(unsigned x)
{ m_estimateNValue->setValue(x); }
//:~