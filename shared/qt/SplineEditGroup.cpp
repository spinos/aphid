/*
 *  SplineEditGroup.cpp
 *
 *  spline edit with name  
 *
 *  Created by jian zhang on 8/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "SplineEditGroup.h"
#include "QSplineEdit.h"

namespace aphid {

SplineEditGroup::SplineEditGroup(const QString& labelName, QWidget *parent) : QGroupBox(parent)
{
	m_lab = new QLabel(labelName);
	m_edit = new QSplineEdit;
	
	QVBoxLayout * yaLayout = new QVBoxLayout;
    yaLayout->addWidget(m_lab);
	yaLayout->addWidget(m_edit);
	yaLayout->setContentsMargins(0, 0, 0, 0);
	
    setLayout(yaLayout);
	
	connect(m_edit, SIGNAL(valueChanged(QPointF)),
            this, SLOT(recvEditValue(QPointF)));
			
	connect(m_edit, SIGNAL(leftControlChanged(QPointF)),
            this, SLOT(recvEditLleftControl(QPointF)));
			
	connect(m_edit, SIGNAL(rightControlChanged(QPointF)),
            this, SLOT(recvEditRightControl(QPointF)));
}

void SplineEditGroup::setNameId(int x)
{ m_nameId = x; }

const int& SplineEditGroup::nameId() const
{ return m_nameId; }

void SplineEditGroup::setSplineValue(const float* x)
{ m_edit->setValue(x); }

void SplineEditGroup::setSplineCv0(const float* x)
{ m_edit->setCv0(x); }

void SplineEditGroup::setSplineCv1(const float* x)
{ m_edit->setCv1(x); }

void SplineEditGroup::recvEditValue(QPointF p)
{ 
	QPair<int, QPointF> val;
	val.first = m_nameId;
	val.second = p;
	emit valueChanged(val); 
}
void SplineEditGroup::recvEditLleftControl(QPointF p)
{ 
	QPair<int, QPointF> val;
	val.first = m_nameId;
	val.second = p;
	emit leftControlChanged(val); 
}

void SplineEditGroup::recvEditRightControl(QPointF p)
{ 
	QPair<int, QPointF> val;
	val.first = m_nameId;
	val.second = p;
	emit rightControlChanged(val); 
}

}