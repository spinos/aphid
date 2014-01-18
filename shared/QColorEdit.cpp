/*
 *  QColorEdit.cpp
 *  spinboxdelegate
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "QColorEdit.h"

QColorEdit::QColorEdit(QColor color, const QModelIndex & idx, QWidget * parent) : QModelEdit(idx, parent)
{
	m_color = color;
	QPalette palette;
	palette.setColor(QPalette::Base, m_color);
	setPalette(palette);
	setFocusPolicy(Qt::NoFocus);
}

void QColorEdit::setColor(QColor color)
{
	m_color = color;
}

QColor QColorEdit::color() const 
{
	return m_color;
}

QColor QColorEdit::pickColor()
{
	QColorDialog dialog(m_color);
	dialog.setOption(QColorDialog::ShowAlphaChannel, true);
	dialog.move(280, 120);
	if (dialog.exec() == QDialog::Rejected)
		return m_color;
	QColor newColor = dialog.selectedColor().rgb();
	setColor(newColor);
	emit editingFinished();
	return m_color;
}
