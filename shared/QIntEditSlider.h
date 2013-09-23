/*
 *  QIntEditSlider.h
 *  mallard
 *
 *  Created by jian zhang on 9/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef Q_INT_EDIT_SLIDER_H
#define Q_INT_EDIT_SLIDER_H

#include <QWidget>

QT_BEGIN_NAMESPACE
class QLabel;
class QLineEdit;
class QSlider;
class QIntValidator;
QT_END_NAMESPACE

class QIntEditSlider : public QWidget 
{
	Q_OBJECT
public:
	QIntEditSlider(const QString & name, QWidget *parent = 0);
	
	void setLimit(int bottom, int top);
	void setValue(int x);
	int value() const;
	
private slots:
	void setEditValue(int x);
	void validateEditValue();
	
signals:
	void valueChanged(int x);
	
private:
	void updateSlider(int x);

private:
	QLabel * m_label;
	QLineEdit * m_edit;
	QSlider * m_slider;
	QIntValidator * m_validate;
	
	int m_bottomValue, m_topValue;
};

#endif