/*
 *  BrushControl.h
 *  mallard
 *
 *  Created by jian zhang on 9/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef BRUSH_CONTROL_H
#define BRUSH_CONTROL_H

#include <QDialog>

QT_BEGIN_NAMESPACE
class QGroupBox;
class QStackedLayout;
class QCheckBox;
QT_END_NAMESPACE

class QIntEditSlider;
class QDoubleEditSlider;

class BrushControl : public QDialog
{
    Q_OBJECT

public:
    BrushControl(QWidget *parent = 0);
	
	QWidget * numSamplesWidget();
	QWidget * radiusWidget();
	QWidget * pitchWidget();
	QWidget * floodRegionWidget();
	QWidget * eraseRegionWidget();

public slots:
	void receiveToolContext(int c);
	
private slots:
	void sendBrushRadius(double d);
	void sendBrushStrength(double d);
signals:
	void brushRadiusChanged(double d);
	void brushStrengthChanged(double d);
private:
	void createGroup();
	void eraseGroup();
	void combGroup();
	void scaleGroup();
	void bendGroup();
	void smoothGroup();
	
private:
	QStackedLayout * stackLayout;
	QGroupBox * controlsGroupC;
	QDoubleEditSlider * m_radiusValueC;
	QDoubleEditSlider * m_pitchValueC;
	QIntEditSlider * m_numSampleValueC;
	QDoubleEditSlider * m_createStrengthValue;
	QCheckBox * m_floodAreaCheck;
	QGroupBox * controlsGroupE;
	QDoubleEditSlider * m_radiusValueE;
	QCheckBox * m_eraseAreaCheck;
	QDoubleEditSlider * m_eraseStrengthValue;
	QGroupBox * controlsGroupB;
	QDoubleEditSlider * m_radiusValueB;
	QGroupBox * controlsGroupS;
	QDoubleEditSlider * m_radiusValueS;
	QGroupBox * controlsGroupD;
	QDoubleEditSlider * m_radiusValueD;
	QGroupBox * controlsGroupDeintersect;
	QDoubleEditSlider * m_radiusValueDeintersect;
	QDoubleEditSlider * m_smoothDirection;
};
#endif