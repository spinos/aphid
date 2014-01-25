/*
 *  QColorBox.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/26/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "QColorBox.h"

QColorBox::QColorBox(QWidget *parent) : QWidget(parent) { setMinimumSize(80, 24); }

void QColorBox::setColor(QColor c) { m_color = c; update(); }
QColor QColorBox::color() const { return m_color; }

void QColorBox::paintEvent( QPaintEvent * )
{
	QPainter painter(this);
	painter.fillRect(rect(), m_color);
}

void QColorBox::changeValue(int x)
{
	int h = m_color.hue();
	int s = m_color.saturation();
	m_color.setHsv(h, s, x);
	update();
}

void QColorBox::mousePressEvent(QMouseEvent * )
{
	chooseColor();
}

void QColorBox::chooseColor()
{
	QColorDialog dialog(m_color);
	dialog.setOption(QColorDialog::ShowAlphaChannel, true);
	dialog.move(280, 120);
	if (dialog.exec() == QDialog::Rejected)
		return;
	QColor newColor = dialog.selectedColor().rgb();
	setColor(newColor);
	emit colorChanged(m_color);
	return;
}