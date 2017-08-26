/*
 *  IntEditGroup.cpp
 *  
 *  int field with label, name_id
 *
 *  Created by jian zhang on 9/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "IntEditGroup.h"

namespace aphid {

IntEditGroup::IntEditGroup(const QString & name, int numFields,
	QWidget *parent)
    : QWidget(parent)
{
	m_label = new QLabel(name);
	m_label->setMinimumWidth(100);
	
	m_numFields = numFields;
	
	m_validate = new QIntValidator;
	
	for(int i=0;i<m_numFields;++i) {
		m_edit[i] = new QLineEdit;
		m_edit[i]->setMaximumWidth(80);
		m_edit[i]->setValidator(m_validate);
	
		m_leftBtn[i] = new QPushButton;
		m_leftBtn[i]->setMaximumWidth(22);
		m_leftBtn[i]->setMaximumHeight(22);
		m_leftBtn[i]->setIcon(QIcon(":icons/left_spin.png"));
		m_rightBtn[i] = new QPushButton;
		m_rightBtn[i]->setMaximumWidth(22);
		m_rightBtn[i]->setMaximumHeight(22);
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
	
	setLimit(0, 10);
	
	for(int i=0;i<m_numFields;++i) {
	
		if(i==0) {
			connect(m_edit[i], SIGNAL(returnPressed()),
					this, SLOT(validateEditValue0()));
				
			connect(m_leftBtn[i], SIGNAL(pressed()),
					this, SLOT(spinValue0Down()));
					
			connect(m_rightBtn[i], SIGNAL(pressed()),
					this, SLOT(spinValue0Up()));
					
		} else if (i == 1) {
			connect(m_edit[i], SIGNAL(returnPressed()),
					this, SLOT(validateEditValue1()));
				
			connect(m_leftBtn[i], SIGNAL(pressed()),
					this, SLOT(spinValue1Down()));
					
			connect(m_rightBtn[i], SIGNAL(pressed()),
					this, SLOT(spinValue1Up()));
					
		} else {
			connect(m_edit[i], SIGNAL(returnPressed()),
					this, SLOT(validateEditValue2()));
				
			connect(m_leftBtn[i], SIGNAL(pressed()),
					this, SLOT(spinValue2Down()));
					
			connect(m_rightBtn[i], SIGNAL(pressed()),
					this, SLOT(spinValue2Up()));
					
		}
	}
	
}

void IntEditGroup::setValue0(int x)
{
	QString t;
	t.setNum(x);
	m_edit[0]->setText(t);
}

int IntEditGroup::value0() const
{
	return m_edit[0]->text().toInt();
}

void IntEditGroup::setValue1(int x)
{
	QString t;
	t.setNum(x);
	m_edit[1]->setText(t);
}

int IntEditGroup::value1() const
{
	return m_edit[1]->text().toInt();
}

void IntEditGroup::setValue2(int x)
{
	QString t;
	t.setNum(x);
	m_edit[2]->setText(t);
}

int IntEditGroup::value2() const
{
	return m_edit[2]->text().toInt();
}

void IntEditGroup::setLimit(int bottom, int top)
{
	m_bottomValue = bottom; 
	m_topValue = top;
}

void IntEditGroup::validateEditValue0()
{
	int x = value0();
	if(x < m_bottomValue) x = m_bottomValue;
	else if(x > m_topValue) x = m_topValue;
	setValue0(x);
	sendValues();
}

void IntEditGroup::validateEditValue1()
{
	int x = value1();
	if(x < m_bottomValue) x = m_bottomValue;
	else if(x > m_topValue) x = m_topValue;
	setValue1(x);
	sendValues();
}

void IntEditGroup::validateEditValue2()
{
	int x = value2();
	if(x < m_bottomValue) x = m_bottomValue;
	else if(x > m_topValue) x = m_topValue;
	setValue2(x);
	sendValues();
}

void IntEditGroup::sendValues()
{
	QPair<int, QVector<int> > val;
	val.first = m_nameId;
	
	QVector<int> vec;
	for(int i=0;i<m_numFields;++i) {
		int vi = m_edit[i]->text().toInt();
		vec<<vi;
	}
	val.second = vec;

	emit valueChanged2(val);
}

void IntEditGroup::spinValue0Up()
{
	int x = value0() + 1;
	if(x <= m_topValue) {
		setValue0(x);
		sendValues();
	}
}

void IntEditGroup::spinValue0Down()
{
	int x = value0() - 1;
	if(x >= m_bottomValue){
		setValue0(x);
		sendValues();
	}
}

void IntEditGroup::spinValue1Up()
{
	int x = value1() + 1;
	if(x <= m_topValue) {
		setValue1(x);
		sendValues();
	}
}

void IntEditGroup::spinValue1Down()
{
	int x = value1() - 1;
	if(x >= m_bottomValue){
		setValue1(x);
		sendValues();
	}
}

void IntEditGroup::spinValue2Up()
{
	int x = value2() + 1;
	if(x <= m_topValue) {
		setValue2(x);
		sendValues();
	}
}

void IntEditGroup::spinValue2Down()
{
	int x = value2() - 1;
	if(x >= m_bottomValue){
		setValue2(x);
		sendValues();
	}
}

void IntEditGroup::setValues(const int* v) 
{
	QString t;
	for(int i=0;i<m_numFields;++i) {
		t.setNum(v[i]);
		m_edit[i]->setText(t);
	}
}

void IntEditGroup::getValues(int* v) const
{
	for(int i=0;i<m_numFields;++i) {
		v[i] = m_edit[i]->text().toInt();
	}
}

void IntEditGroup::setNameId(int x)
{ m_nameId = x; }

const int& IntEditGroup::nameId() const
{ return m_nameId; }

}