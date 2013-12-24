/*
 *  BarbControl.h
 *  mallard
 *
 *  Created by jian zhang on 12/25/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <QWidget>

QT_BEGIN_NAMESPACE
class QGroupBox;
class QStackedLayout;
class QCheckBox;
QT_END_NAMESPACE

class QIntEditSlider;
class QDoubleEditSlider;
class QDoubleEditSlider;
class BarbControl : public QWidget
{
    Q_OBJECT

public:
	BarbControl(QWidget *parent = 0);

public slots:
	
private slots:
	void sendSeed(int s);
	void sendNumSeparate(int n);
	void sendSeparateStrength(double k);
signals:
	void seedChanged(int s);
	void numSeparateChanged(int n);
	void separateStrengthChanged(double k);
private:
	
	
private:
	QIntEditSlider * m_seedValue;
	QIntEditSlider * m_separateCountValue;
	QDoubleEditSlider * m_separateWeightValue;
};
