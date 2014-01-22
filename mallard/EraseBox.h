/*
 *  EraseBox.h
 *  mallard
 *
 *  Created by jian zhang on 1/23/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <QGroupBox>

QT_BEGIN_NAMESPACE
class QCheckBox;
QT_END_NAMESPACE

class QDoubleEditSlider;
class QIntEditSlider;
class EraseBox : public QGroupBox {
	Q_OBJECT

public:
	EraseBox(QWidget * parent = 0);
	
	double radius() const;
	double strength() const;
	
signals:
	void radiusChanged(double x);
	void strengthChanged(double x);
	void eraseRegionChanged(int x);
	
private slots:
	void sendRadius(double x);
	void sendStrength(double x);
	void sendEraseRegion(int x);
	
private:
	QDoubleEditSlider * m_radiusValue;
	QDoubleEditSlider * m_strengthValue;
	QCheckBox * m_areaCheck;
};