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

IntEditGroup::IntEditGroup(const QString & name, QWidget *parent)
    : QWidget(parent)
{
	m_label = new QLabel(name);
	m_label->setMinimumWidth(100);
	m_edit = new QLineEdit;
	m_edit->setMaximumWidth(80);
	m_validate = new QIntValidator;
	
	m_edit->setValidator(m_validate);
	
	m_leftBtn = new QPushButton;
	m_leftBtn->setMaximumWidth(22);
	m_leftBtn->setMaximumHeight(22);
	m_leftBtn->setIcon(QIcon(":icons/left_spin.png"));
	m_rightBtn = new QPushButton;
	m_rightBtn->setMaximumWidth(22);
	m_rightBtn->setMaximumHeight(22);
	m_rightBtn->setIcon(QIcon(":icons/right_spin.png"));
	
	QHBoxLayout * layout = new QHBoxLayout;
	layout->addWidget(m_label);
	layout->addWidget(m_edit);
	layout->addWidget(m_leftBtn);
	layout->addWidget(m_rightBtn);
	layout->addStretch(1);
	layout->setContentsMargins(0, 0, 0, 0);
	setLayout(layout);
	
	setLimit(0, 10);
	setValue(0);
	
	connect(m_edit, SIGNAL(returnPressed()),
            this, SLOT(validateEditValue()));
		
	connect(m_leftBtn, SIGNAL(pressed()),
            this, SLOT(spinValueDown()));
			
	connect(m_rightBtn, SIGNAL(pressed()),
            this, SLOT(spinValueUp()));
}

void IntEditGroup::setLimit(int bottom, int top)
{
	m_bottomValue = bottom; 
	m_topValue = top;
}

void IntEditGroup::setValue(int x)
{
	setEditValue(x);
}

int IntEditGroup::value() const
{
	return m_edit->text().toInt();
}

void IntEditGroup::validateEditValue()
{
	int x = value();
	if(x < m_bottomValue) x = m_bottomValue;
	else if(x > m_topValue) x = m_topValue;
	setValue(x);
}

void IntEditGroup::setEditValue(int x)
{
	QString t;
	t.setNum(x);
	m_edit->setText(t);
	QPair<int, int> val;
	val.first = m_nameId;
	val.second = x;

	emit valueChanged2(val);
}

void IntEditGroup::spinValueUp()
{
	int x = value() + 1;
	if(x <= m_topValue)
		setEditValue(x);
}

void IntEditGroup::spinValueDown()
{
	int x = value() - 1;
	if(x >= m_bottomValue)
		setEditValue(x);
}

void IntEditGroup::setNameId(int x)
{ m_nameId = x; }

const int& IntEditGroup::nameId() const
{ return m_nameId; }

}