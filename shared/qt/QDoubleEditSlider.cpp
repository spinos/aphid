/*
 *  QDoubleEditSlider.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/24/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "QDoubleEditSlider.h"

namespace aphid {

QDoubleEditSlider::QDoubleEditSlider(const QString & name, QWidget *parent)
    : QWidget(parent)
{
	m_label = new QLabel(name);
	m_label->setMinimumWidth(100);
	m_edit = new QLineEdit;
	m_edit->setMaximumWidth(80);
	m_slider = new QSlider(Qt::Horizontal);
	m_validate = new QDoubleValidator(this);
	
	m_edit->setValidator(m_validate);
	m_slider->setRange(0, 100);
	
	QHBoxLayout * layout = new QHBoxLayout;
	layout->addWidget(m_label);
	layout->addWidget(m_edit);
	layout->addWidget(m_slider);
	layout->setStretch(2, 1);
	layout->setContentsMargins(0, 0, 0, 0);
	setLayout(layout);
	
	setLimit(0.1, 10.0);
	setValue(5.0);
	
	connect(m_edit, SIGNAL(returnPressed()),
            this, SLOT(validateEditValue()));
			
	connect(m_slider, SIGNAL(valueChanged(int)),
            this, SLOT(convertEditValue(int)));
}

void QDoubleEditSlider::setLimit(double bottom, double top)
{
	m_bottomValue = bottom; 
	m_topValue = top;
}

void QDoubleEditSlider::setValue(double x)
{
	setEditValue(x);
    disconnect(m_slider, SIGNAL(valueChanged(int)),
            this, SLOT(convertEditValue(int)));
	updateSlider(x);
    connect(m_slider, SIGNAL(valueChanged(int)),
            this, SLOT(convertEditValue(int)));
}

double QDoubleEditSlider::value() const
{
	return m_edit->text().toDouble();
}

void QDoubleEditSlider::validateEditValue()
{
	double x = value();
	if(x < m_bottomValue) x = m_bottomValue;
	else if(x > m_topValue) x = m_topValue;
	setValue(x);
}

void QDoubleEditSlider::updateSlider(double x)
{
	double slideMin = x / 2;
	double slideMax = x * 2;
	if(slideMin < m_bottomValue) slideMin = m_bottomValue;
	if(slideMax > m_topValue) slideMax = m_topValue;
	if(x == m_bottomValue || x == m_topValue) {
		slideMin = m_bottomValue;
		slideMax = m_topValue;
	}
	
	m_slideMin = slideMin;
	m_slideMax = slideMax;
	
	int i = (x - m_slideMin) / (m_slideMax - m_slideMin) * 100;
	m_slider->setValue(i);
}

void QDoubleEditSlider::setEditValue(double x)
{
	QString t;
	t.setNum(x);
	m_edit->setText(t);
	emit valueChanged(x);
	QPair<int, double> val;
	val.first = m_nameId;
	val.second = x;
	emit valueChanged2(val);
}

void QDoubleEditSlider::convertEditValue(int x)
{
	const double d = m_slideMin + (m_slideMax - m_slideMin) * (double)x / 100.0;
	setEditValue(d);
}

void QDoubleEditSlider::setNameId(int x)
{ m_nameId = x; }

const int& QDoubleEditSlider::nameId() const
{ return m_nameId; }

}