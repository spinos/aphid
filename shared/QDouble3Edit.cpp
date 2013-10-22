/*
 *  QDouble3Edit.cpp
 *  eulerRot
 *
 *  Created by jian zhang on 10/22/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "QDouble3Edit.h"

QDouble3Edit::QDouble3Edit(const QString & name, QWidget *parent)
    : QWidget(parent)
{
	m_label = new QLabel(name);
	m_label->setMinimumWidth(80);
	m_edit0 = new QLineEdit;
	m_edit0->setMinimumWidth(40);
	m_edit0->setMaximumWidth(80);
	m_edit1 = new QLineEdit;
	m_edit1->setMinimumWidth(40);
	m_edit1->setMaximumWidth(80);
	m_edit2 = new QLineEdit;
	m_edit2->setMinimumWidth(40);
	m_edit2->setMaximumWidth(80);
	m_validate = new QDoubleValidator;
	
	m_edit0->setValidator(m_validate);
	m_edit1->setValidator(m_validate);
	m_edit2->setValidator(m_validate);
	
	QHBoxLayout * layout = new QHBoxLayout;
	layout->addWidget(m_label);
	layout->addWidget(m_edit0);
	layout->addWidget(m_edit1);
	layout->addWidget(m_edit2);
	layout->setStretch(0, 1);
	layout->setContentsMargins(0, 0, 0, 0);
	setLayout(layout);
	
	setValue(Vector3F());
	
	connect(m_edit0, SIGNAL(returnPressed()),
            this, SLOT(validateEditValue()));
			
	connect(m_edit1, SIGNAL(returnPressed()),
            this, SLOT(validateEditValue()));
			
	connect(m_edit2, SIGNAL(returnPressed()),
            this, SLOT(validateEditValue()));
}

void QDouble3Edit::setValue(const Vector3F & v) 
{
	QString t;
	t.setNum(v.x);
	m_edit0->setText(t);
	t.setNum(v.y);
	m_edit1->setText(t);
	t.setNum(v.z);
	m_edit2->setText(t);
}

Vector3F QDouble3Edit::value() const
{
	return Vector3F(m_edit0->text().toDouble(), m_edit1->text().toDouble(), m_edit2->text().toDouble());
}

void QDouble3Edit::validateEditValue()
{
	emit valueChanged(value());
}

void QDouble3Edit::setDOF(const Float3 & dof)
{
	if(dof.x == 0.f) m_edit0->setEnabled(0);
	else m_edit0->setEnabled(1);
	if(dof.y == 0.f) m_edit1->setEnabled(0);
	else m_edit1->setEnabled(1);
	if(dof.z == 0.f) m_edit2->setEnabled(0);
	else m_edit2->setEnabled(1);
}
