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

private slots:
	
signals:
	
private:
	QGroupBox *controlsGroup;
	QDoubleEditSlider * m_radiusValue;
	QDoubleEditSlider * m_pitchValue;
	QIntEditSlider * m_numSampleValue;
};

#endif