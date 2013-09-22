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
QT_END_NAMESPACE

class BrushControl : public QDialog
{
    Q_OBJECT

public:
    BrushControl(QWidget *parent = 0);

private slots:
    void radiusSliderChanged(int value);
	void pitchSliderChanged(int value);
private:
	QGroupBox *controlsGroup;
	QLabel *radius;
	QLineEdit *radiusValue;
	QSlider *radiusSlider;
	QLabel *pitch;
	QLineEdit *pitchValue;
	QSlider *pitchSlider;
};

#endif