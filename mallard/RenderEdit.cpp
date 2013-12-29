/*
 *  RenderEdit.cpp
 *  mallard
 *
 *  Created by jian zhang on 12/29/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "RenderEdit.h"
#include "ImageView.h"

RenderEdit::RenderEdit(QWidget *parent)
    : QDialog(parent)
{
	m_view = new ImageView(this);
	
	QVBoxLayout *layout = new QVBoxLayout;
	//layout->addWidget(l);
	layout->addWidget(m_view);
	setLayout(layout);
	setWindowTitle(tr("Render View"));
	
	setContentsMargins(0, 0, 0, 0);
	layout->setContentsMargins(0, 0, 0, 0);
	
	//connect(m_control, SIGNAL(seedChanged(int)), m_view, SLOT(receiveSeed(int)));
	//connect(m_control, SIGNAL(numSeparateChanged(int)), m_view, SLOT(receiveNumSeparate(int)));
	//connect(m_control, SIGNAL(separateStrengthChanged(double)), m_view, SLOT(receiveSeparateStrength(double)));
	//connect(m_control, SIGNAL(fuzzyChanged(double)), m_view, SLOT(receiveFuzzy(double)));
	//connect(m_control, SIGNAL(gridShaftChanged(int)), m_view, SLOT(receiveGridShaft(int)));
	//connect(m_control, SIGNAL(gridBarbChanged(int)), m_view, SLOT(receiveGridBarb(int)));
}