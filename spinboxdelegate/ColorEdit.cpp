/*
 *  ColorEdit.cpp
 *  spinboxdelegate
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "ColorEdit.h"

ColorEdit::ColorEdit(QWidget * parent) : QWidget(parent)
{
	QHBoxLayout *layout = new QHBoxLayout;
    setLayout(layout);
    layout->setContentsMargins(0, 0, 0, 0);

    m_button = new QFrame;
    QPalette palette = m_button->palette();
    palette.setColor(QPalette::Window, QColor(m_color));
    m_button->setPalette(palette);
    m_button->setAutoFillBackground(true);
    m_button->setMinimumSize(32, 0);
    m_button->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
    m_button->setFrameStyle(QFrame::StyledPanel | QFrame::Sunken);
    layout->addWidget(m_button);
}

void ColorEdit::setColor(QColor color)
{
	qDebug()<<"set col"<<color;
	m_color = color;
}

QColor ColorEdit::color() const 
{
	return m_color;
}

void ColorEdit::mousePressEvent(QMouseEvent *event)
{
	if (event->button() == Qt::LeftButton) {
        QColor color(m_color);
        QColorDialog dialog(color, 0);
        dialog.setOption(QColorDialog::ShowAlphaChannel, true);
        dialog.move(280, 120);
        if (dialog.exec() == QDialog::Rejected)
            return;
        QColor newColor = dialog.selectedColor().rgb();
		qDebug()<<"chose col"<<newColor;
        setColor(newColor);
    }
}