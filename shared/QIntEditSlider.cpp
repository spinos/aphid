/*
 *  QIntEditSlider.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "QIntEditSlider.h"

QIntEditSlider::QIntEditSlider(const QString & name, QWidget *parent)
    : QWidget(parent)
{
	m_label = new QLabel(name);
	m_edit = new QLineEdit;
	m_slider = new QSlider(Qt::Horizontal);
	m_validate = new QIntValidator;
	
	m_edit->setValidator(m_validate);
	
	QHBoxLayout * layout = new QHBoxLayout;
	layout->addWidget(m_label);
	layout->addWidget(m_edit);
	layout->addWidget(m_slider);
	layout->setStretch(2, 1);
	setLayout(layout);
	
	setLimit(0, 10);
	setValue(5);
	updateSlider(5);
	
	connect(m_edit, SIGNAL(returnPressed()),
            this, SLOT(validateEditValue()));
			
	connect(m_slider, SIGNAL(valueChanged(int)),
            this, SLOT(setEditValue(int)));
}

void QIntEditSlider::setLimit(int bottom, int top)
{
	m_bottomValue = bottom; 
	m_topValue = top;
}

void QIntEditSlider::setValue(int x)
{
	setEditValue(x);
	updateSlider(x);
}

int QIntEditSlider::value() const
{
	return m_edit->text().toInt();
}

void QIntEditSlider::validateEditValue()
{
	int x = value();
	if(x < m_bottomValue) x = m_bottomValue;
	else if(x > m_topValue) x = m_topValue;
	setValue(x);
}

void QIntEditSlider::updateSlider(int x)
{
	int slideMin = x / 3;
	int slideMax = x * 3;
	if(slideMin < m_bottomValue) slideMin = m_bottomValue;
	if(slideMax > m_topValue) slideMax = m_topValue;
	m_slider->setRange(slideMin, slideMax);
	m_slider->setValue(x);
}

void QIntEditSlider::setEditValue(int x)
{
	QString t;
	t.setNum(x);
	m_edit->setText(t);
	emit valueChanged(x);
}

