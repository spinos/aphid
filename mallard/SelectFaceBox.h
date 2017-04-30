/*
 *  SelectFaceBox.h
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
class SelectFaceBox : public QGroupBox {
	Q_OBJECT

public:
	SelectFaceBox(QWidget * parent = 0);
	
	double radius() const;
	int twoSided() const;
	
signals:
	void radiusChanged(double x);
	void twoSidedChanged(int);
	
private slots:
	void sendRadius(double x);
	void sendTwoSided(int);
	
private:
	QDoubleEditSlider * m_radiusValue;
	QCheckBox * m_twoSidedCheck;
};