/*
 *  QColorEditSlider.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/26/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "QColorEditSlider.h"
#include "QColorBox.h"

QColorEditSlider::QColorEditSlider(const QString & name, QWidget *parent)
    : QWidget(parent)
{
	m_label = new QLabel(name);
	m_label->setMinimumWidth(100);
	
	m_edit = new QColorBox;
	
	m_slider = new QSlider(Qt::Horizontal);
	m_slider->setRange(0, 255);
	
	QHBoxLayout * layout = new QHBoxLayout;
	layout->addWidget(m_label);
	layout->addWidget(m_edit);
	layout->addWidget(m_slider);
	layout->setStretch(2, 1);
	layout->setContentsMargins(0, 0, 0, 0);
	setLayout(layout);
	
	connect(m_edit, SIGNAL(colorChanged(QColor)),
            this, SLOT(updateSlider(QColor)));
			
	connect(m_slider, SIGNAL(valueChanged(int)),
            m_edit, SLOT(changeValue(int)));
			
	connect(m_slider, SIGNAL(valueChanged(int)),
            this, SLOT(sendValue(int)));
}

void QColorEditSlider::setValue(QColor c)
{
	m_edit->setColor(c);
	updateSlider(c);
}

QColor QColorEditSlider::value() const
{
	return m_edit->color();
}

void QColorEditSlider::updateSlider(QColor c)
{
	int h = c.value();
	
	m_slider->setValue(h);
	emit valueChanged(c);
}

void QColorEditSlider::sendValue(int) { emit valueChanged(value()); }
