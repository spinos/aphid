/*
 *  Cooktorrance.h
 *  
 *
 *  Created by jian zhang on 10/2/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <BRDFProgram.h>

#include <QWidget>

QT_BEGIN_NAMESPACE
class QSlider;
class QLabel;
class QLineEdit;
class QGroupBox;
class QCheckBox;
QT_END_NAMESPACE

class Cooktorrance : public QWidget, public BRDFProgram
{
	Q_OBJECT

public:
    Cooktorrance();
	
	virtual void run(CUDABuffer * buffer, BaseMesh * mesh);
	
public slots:
	void setMValue(int value);
	void setF0Value(int value);
	void setIncfValue(int value);
	void setIncgValue(int value);

private:
	QGroupBox *controlsGroup;
	QLabel *mName;
	QLineEdit *mValue;
	QSlider *mSlider;
	QLabel *f0Name;
	QLineEdit *f0Value;
	QSlider *f0Slider;
	QLabel *incfName;
	QCheckBox *incfControl;
	QLabel *incgName;
	QCheckBox *incgControl;
	
	float _m, _f0;
	int _includeF, _includeG;
};