/*
 *  SynthControl.h
 *  cudafem
 *
 *  Created by jian zhang on Mon Jul 27 18:34:17 CST 2015 
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef DTHF_SYNTH_CONTROL_H
#define DTHF_SYNTH_CONTROL_H

#include <QDialog>

QT_BEGIN_NAMESPACE
class QGroupBox;
class QCheckBox;
class QLabel;
QT_END_NAMESPACE

namespace aphid {

class QDoubleEditSlider;
class ExrImage;
class NavigatorWidget;

}

class SynthControl : public QDialog
{
    Q_OBJECT

public:
    SynthControl(const aphid::ExrImage * img,
				QWidget *parent = 0);
	
public slots:
	
private slots:
	void sendA(double x);
	void sendB(double x);
	void sendC(double x);
	void sendD(double x);
	
signals:
	void l0ScaleChanged(double a);
    void l1ScaleChanged(double a);
    void l2ScaleChanged(double a);
    void l3ScaleChanged(double a);
    
private:
	aphid::NavigatorWidget * m_navigator;
    aphid::QDoubleEditSlider * m_aValue;
	aphid::QDoubleEditSlider * m_bValue;
	aphid::QDoubleEditSlider * m_cValue;
	aphid::QDoubleEditSlider * m_dValue;
	QGroupBox * S1Grp;
		
};
#endif