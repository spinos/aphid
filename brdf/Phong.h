/*
 *  Phong.h
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

class Phong : public QWidget, public BRDFProgram
{
	Q_OBJECT

public:
    Phong();
	
	virtual void run(CUDABuffer * buffer, HemisphereMesh * mesh);
	
public slots:
	void setExposureValue(int value);
	void setDivideByNdoLValue(int value);

private:
	QGroupBox *controlsGroup;
	QLabel *exposureName;
	QLineEdit *exposureValue;
	QSlider *exposureSlider;
	QLabel *divideByNdoLName;
	QCheckBox *divideByNdoLControl;
	
	float _exposure;
	int _divideByNdoL;
};