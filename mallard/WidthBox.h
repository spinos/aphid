/*
 *  WidthBox.h
 *  mallard
 *
 *  Created by jian zhang on 1/30/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <QGroupBox>
class QDoubleEditSlider;
class WidthBox : public QGroupBox {
	Q_OBJECT

public:
	WidthBox(QWidget * parent = 0);
	
	double radius() const;
	
signals:
	void radiusChanged(double x);
	
private slots:
	void sendRadius(double x);
	
private:
	QDoubleEditSlider * m_radiusValue;
};