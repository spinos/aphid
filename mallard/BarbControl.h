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
#include <FeatherExample.h>
QT_BEGIN_NAMESPACE
class QGroupBox;
class QStackedLayout;
class QCheckBox;
QT_END_NAMESPACE

class QIntEditSlider;
class QDoubleEditSlider;
class QDoubleEditSlider;
class BarbControl : public QWidget, public FeatherExample
{
    Q_OBJECT

public:
	BarbControl(QWidget *parent = 0);

public slots:
	void receiveSelectionChanged();
private slots:
	void sendSeed(int s);
	void sendNumSeparate(int n);
	void sendSeparateStrength(double k);
	void sendFuzzy(double f);
	void sendGridShaft(int g);
	void sendGridBarb(int g);
	void sendLod(double l);
signals:
	void shapeChanged();
private:
	
	
private:
	QIntEditSlider * m_seedValue;
	QIntEditSlider * m_gridShaftValue;
	QIntEditSlider * m_gridBarbValue;
	QIntEditSlider * m_separateCountValue;
	QDoubleEditSlider * m_separateWeightValue;
	QDoubleEditSlider * m_fuzzyValue;
	QDoubleEditSlider * m_lodValue;
};
