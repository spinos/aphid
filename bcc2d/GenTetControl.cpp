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
	m_estimateNValue->setLimit(100.0, 500000.0);
	m_estimateNValue->setValue(2000.0);
	
	QGroupBox * estimateNGrp = new QGroupBox;
    QHBoxLayout * yLayout = new QHBoxLayout;
	yLayout->addWidget(m_estimateNValue);
	
    estimateNGrp->setLayout(yLayout);
    
    m_patchMethodChooser = new QComboBox(this);
    m_patchMethodChooser->addItem(tr("Block"));
    m_patchMethodChooser->addItem(tr("Single octahedron"));
    
    QGroupBox * patchMethodGrp = new QGroupBox;
    QHBoxLayout * patchMethodLayout = new QHBoxLayout;
    QLabel * patchMethodLabel = new QLabel(tr("Patch generation method"));
	patchMethodLayout->addWidget(patchMethodLabel);
    patchMethodLayout->addWidget(m_patchMethodChooser);
	patchMethodLayout->setStretch(1, 1);
    patchMethodGrp->setLayout(patchMethodLayout);
    
    m_rebuildAct = new QPushButton(tr("Rebuild tetrahedron mesh"));
	
    QVBoxLayout *layout = new QVBoxLayout;
	layout->addWidget(estimateNGrp);
    layout->addWidget(patchMethodGrp);
    layout->addWidget(m_rebuildAct);
	layout->setStretch(3, 1);
	layout->setSpacing(4);
	
	setLayout(layout);
    
    connect(m_rebuildAct, SIGNAL(clicked()), this, SLOT(sendRebuild()));
    connect(m_patchMethodChooser, SIGNAL(currentIndexChanged(int)), 
            this, SLOT(sendPatchMethod(int)));
}

void GenTetControl::sendRebuild()
{ 
    double n = m_estimateNValue->value();
    emit rebuildTet(n); 
}

void GenTetControl::receiveEstimatedN(unsigned x)
{ m_estimateNValue->setValue(x); }

void GenTetControl::sendPatchMethod(int x)
{ emit patchMethodChanged(x); }
//:~