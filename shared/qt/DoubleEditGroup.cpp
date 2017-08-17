/*
 *  DoubleEditGroup.cpp
 *
 *  Created by jian zhang on 10/22/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "DoubleEditGroup.h"

DoubleEditGroup::DoubleEditGroup(const QString & name, int numFields,
QWidget *parent)
    : QGroupBox(parent)
{
	m_label = new QLabel(name);
	m_label->setMinimumWidth(100);
	
	m_validate = new QDoubleValidator;
	
	m_numFields = numFields;
	
	for(int i=0;i<m_numFields;++i) {
		m_edit[i] = new QLineEdit;
		m_edit[i]->setMaximumWidth(80);
		m_edit[i]->setValidator(m_validate);
		
		m_leftBtn[i] = new QPushButton;
		m_leftBtn[i]->setMaximumWidth(20);
		m_leftBtn[i]->setMaximumHeight(20);
		m_leftBtn[i]->setIcon(QIcon(":icons/left_spin.png"));
		m_rightBtn[i] = new QPushButton;
		m_rightBtn[i]->setMaximumWidth(20);
		m_rightBtn[i]->setMaximumHeight(20);
		m_rightBtn[i]->setIcon(QIcon(":icons/right_spin.png"));
	}

	QGridLayout * layout = new QGridLayout;
	layout->addWidget(m_label, 0, 0);
	
	for(int i=0;i<m_numFields;++i) {
		layout->addWidget(m_edit[i], i, 1);
		layout->addWidget(m_leftBtn[i], i, 2);
		layout->addWidget(m_rightBtn[i], i, 3);
	}
	
	layout->setColumnStretch(4, 1);
	layout->setContentsMargins(0, 0, 0, 0);
	setLayout(layout);
	
	for(int i=0;i<m_numFields;++i) {
		connect(m_edit[i], SIGNAL(returnPressed()),
            this, SLOT(sendValues()));
			
		if(i==0) {
			connect(m_leftBtn[i], SIGNAL(pressed()),
            this, SLOT(spinValue0Down()));
			
			connect(m_rightBtn[i], SIGNAL(pressed()),
            this, SLOT(spinValue0Up()));
		} else if(i==1) {
			connect(m_leftBtn[i], SIGNAL(pressed()),
            this, SLOT(spinValue1Down()));
			
			connect(m_rightBtn[i], SIGNAL(pressed()),
            this, SLOT(spinValue1Up()));
		} else {
			connect(m_leftBtn[i], SIGNAL(pressed()),
            this, SLOT(spinValue2Down()));
			
			connect(m_rightBtn[i], SIGNAL(pressed()),
            this, SLOT(spinValue2Up()));
		}
	}
	
}

void DoubleEditGroup::spinValue0Up()
{
	spinValueUp(0);
}

void DoubleEditGroup::spinValue0Down()
{
	spinValueDown(0);
}

void DoubleEditGroup::spinValue1Up()
{
	spinValueUp(1);
}

void DoubleEditGroup::spinValue1Down()
{
	spinValueDown(1);
}

void DoubleEditGroup::spinValue2Up()
{
	spinValueUp(2);
}

void DoubleEditGroup::spinValue2Down()
{
	spinValueDown(2);
}

void DoubleEditGroup::spinValueDown(int i)
{
	float v = (float)m_edit[i]->text().toDouble();
	v -= 0.1f;
	QString t;
	t.setNum(v);
	m_edit[i]->setText(t);
	sendValues();
}

void DoubleEditGroup::spinValueUp(int i)
{
	float v = (float)m_edit[i]->text().toDouble();
	v += 0.1f;
	QString t;
	t.setNum(v);
	m_edit[i]->setText(t);
	sendValues();
}

void DoubleEditGroup::setValues(const float* v) 
{
	QString t;
	for(int i=0;i<m_numFields;++i) {
		t.setNum(v[i]);
		m_edit[i]->setText(t);
	}
}

void DoubleEditGroup::getValues(float* v) const
{
	for(int i=0;i<m_numFields;++i) {
		v[i] = (float)m_edit[i]->text().toDouble();
	}
}

void DoubleEditGroup::sendValues()
{
	QPair<int, QVector<double> > val;
	val.first = m_nameId;
	
	QVector<double> vec;
	for(int i=0;i<m_numFields;++i) {
		double vi = m_edit[i]->text().toDouble();
		vec<<vi;
	}
	val.second = vec;
	
	emit valueChanged2(val);
}

void DoubleEditGroup::setNameId(int x)
{ m_nameId = x; }

const int& DoubleEditGroup::nameId() const
{ return m_nameId; }
