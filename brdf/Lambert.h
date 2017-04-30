/*
 *  Lambert.h
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
QT_END_NAMESPACE

class Lambert : public QWidget, public BRDFProgram
{
	Q_OBJECT

public:
    Lambert();
	
	virtual void run(CUDABuffer * buffer, BaseMesh * mesh);
	
public slots:
	void setReflectanceValue(int value);

private:
	QGroupBox *controlsGroup;
	QLabel *reflectance;
	QLineEdit *reflectanceValue;
	QSlider *slider;
	
	float _reflectance;
};

