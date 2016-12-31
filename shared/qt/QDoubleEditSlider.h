/*
 *  QDoubleEditSlider.h
 *  mallard
 *
 *  Created by jian zhang on 9/24/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef Q_DOUBLE_EDIT_SLIDER_H
#define Q_DOUBLE_EDIT_SLIDER_H

#include <QWidget>

QT_BEGIN_NAMESPACE
class QLabel;
class QLineEdit;
class QSlider;
class QDoubleValidator;
QT_END_NAMESPACE

namespace aphid {

class QDoubleEditSlider : public QWidget 
{
	Q_OBJECT
public:
	QDoubleEditSlider(const QString & name, QWidget *parent = 0);
	
	void setLimit(double bottom, double top);
	void setValue(double x);
	double value() const;
	
private slots:
	void setEditValue(double x);
	void convertEditValue(int x);
	void validateEditValue();
	
signals:
	void valueChanged(double x);
	
private:
	void updateSlider(double x);

private:
	QLabel * m_label;
	QLineEdit * m_edit;
	QSlider * m_slider;
	QDoubleValidator * m_validate;
	
	double m_bottomValue, m_topValue;
	double m_slideMin, m_slideMax;
};

}
#endif