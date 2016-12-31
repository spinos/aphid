/*
 *  QColorEditSlider.h
 *  mallard
 *
 *  Created by jian zhang on 1/26/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <QWidget>

QT_BEGIN_NAMESPACE
class QLabel;
class QSlider;
class QColor;
QT_END_NAMESPACE

class QColorBox;

class QColorEditSlider : public QWidget 
{
	Q_OBJECT
public:
	QColorEditSlider(const QString & name, QWidget *parent = 0);
	
	void setValue(QColor c);
	QColor value() const;
	
private slots:
	void updateSlider(QColor c);
	void sendValue(int x);
signals:
	void valueChanged(QColor c);
	
private:
	

private:
	QLabel * m_label;
	QColorBox * m_edit;
	QSlider * m_slider;
};
