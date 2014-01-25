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
class PaintBox : public QGroupBox {
	Q_OBJECT

public:
	PaintBox(QWidget * parent = 0);
	
	double radius() const;
	
signals:
	void radiusChanged(double x);
	
private slots:
	void sendRadius(double x);
	
private:
	QDoubleEditSlider * m_radiusValue;
};