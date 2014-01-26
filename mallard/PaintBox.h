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
	double dropoff() const;
	double strength() const;
	
signals:
	void radiusChanged(double x);
	void colorChanged(QColor c);
	void dropoffChanged(double x);
	void strengthChanged(double x);
	
private slots:
	void sendRadius(double x);
	void sendColor(QColor c);
	void sendDropoff(double x);
	void sendStrength(double x);
	
private:
	QDoubleEditSlider * m_radiusValue;
	QDoubleEditSlider * m_dropoffValue;
	QDoubleEditSlider * m_strengthValue;
	QColorEditSlider * m_colorValue;
};