/*
 *  PaintBox.h
 *  mallard
 *
 *  Created by jian zhang on 1/26/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <QGroupBox>
class QDoubleEditSlider;
class QColorEditSlider;

class PaintBox : public QGroupBox {
	Q_OBJECT

public:
	PaintBox(QWidget * parent = 0);
	
	double radius() const;
	QColor color() const;
	
signals:
	void radiusChanged(double x);
	void colorChanged(QColor c);
	
private slots:
	void sendRadius(double x);
	void sendColor(QColor c);
	
private:
	QDoubleEditSlider * m_radiusValue;
	QColorEditSlider * m_colorValue;
};