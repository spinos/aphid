/*
 *  QColorEdit.cpp
 *  spinboxdelegate
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "QColorEdit.h"

QColorEdit::QColorEdit(QColor color, QWidget * parent) : QWidget(parent)
{
	m_color = color;qDebug()<<"init col"<<color;
	QHBoxLayout *layout = new QHBoxLayout;
    setLayout(layout);
    layout->setContentsMargins(0, 0, 0, 0);

    m_button = new QFrame;
    QPalette palette = m_button->palette();
    palette.setColor(QPalette::Base, m_color);
    m_button->setPalette(palette);
    m_button->setAutoFillBackground(true);
    m_button->setMinimumSize(parent->size());
    m_button->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
    m_button->setFrameStyle(QFrame::StyledPanel | QFrame::Sunken);
    layout->addWidget(m_button);
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
	return m_color;
}
