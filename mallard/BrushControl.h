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
#include <QQueue>

QT_BEGIN_NAMESPACE
class QComboBox;
class QDialogButtonBox;
class QGridLayout;
class QGroupBox;
class QLabel;
class QPushButton;
class QLineEdit;
class QSlider;
class QDoubleValidator;
class QIntValidator;
QT_END_NAMESPACE

class BrushControl : public QDialog
{
    Q_OBJECT

public:
    BrushControl(QWidget *parent = 0);

private slots:
    void radiusSliderChanged(int value);
	void pitchSliderChanged(int value);
	void maxRadiusEdited();
	void numSamplesEdited();
	
signals:
	void radiusChanged(double c);
	void pitchChanged(double c);
	void numSamplesChanged(int c);
	
private:
	QGroupBox *controlsGroup;
	QLabel * maxR;
	QDoubleValidator * maxRVal;
	QLineEdit *maxRValue;
	QLabel *radius;
	QLineEdit *radiusValue;
	QSlider *radiusSlider;
	QLabel *pitch;
	QLineEdit *pitchValue;
	QSlider *pitchSlider;
	QLabel * numSamples;
	QLineEdit *numSamplesValue;
	QIntValidator * numSamplesVal;
};

#endif