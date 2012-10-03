/*
 *  Ward.h
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

class Ward : public QWidget, public BRDFProgram
{
	Q_OBJECT

public:
    Ward();
	
	virtual void run(CUDABuffer * buffer, HemisphereMesh * mesh);
	
public slots:
	void setAlphaXValue(int value);
	void setAlphaYValue(int value);
	void setAnisotropicValue(int value);

private:
	QGroupBox *controlsGroup;
	QLabel *axName;
	QLineEdit *axValue;
	QSlider *axSlider;
	QLabel *ayName;
	QLineEdit *ayValue;
	QSlider *aySlider;
	QLabel *anisotropicName;
	QCheckBox *anisotropicControl;
	
	float _alphaX, _alphaY;
	int _anisotropic;
};