/*
 *  SuperformulaControl.h
 *  cudafem
 *
 *  Created by jian zhang on Mon Jul 27 18:34:17 CST 2015 
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TTG_SUPERFORMULA_CONTROL_H
#define TTG_SUPERFORMULA_CONTROL_H

#include <QDialog>

QT_BEGIN_NAMESPACE
class QGroupBox;
class QCheckBox;
class QLabel;
QT_END_NAMESPACE

class QDoubleEditSlider;

namespace ttg {

class SuperformulaControl : public QDialog
{
    Q_OBJECT

public:
    SuperformulaControl(QWidget *parent = 0);
	
public slots:
	
private slots:
	void sendA1(double x);
	void sendB1(double x);
	void sendM1(double x);
	void sendN1(double x);
	void sendN2(double x);
	void sendN3(double x);
	
	void sendA2(double x);
	void sendB2(double x);
	void sendM2(double x);
	void sendN21(double x);
	void sendN22(double x);
	void sendN23(double x);
	
signals:
	void a1Changed(double a);
	void b1Changed(double a);
	void m1Changed(double a);
	void n1Changed(double a);
	void n2Changed(double a);
	void n3Changed(double a);
	
	void a2Changed(double a);
	void b2Changed(double a);
	void m2Changed(double a);
	void n21Changed(double a);
	void n22Changed(double a);
	void n23Changed(double a);
    
private:
    QDoubleEditSlider * m_a1Value;
	QDoubleEditSlider * m_b1Value;
	QDoubleEditSlider * m_m1Value;
	QDoubleEditSlider * m_n1Value;
	QDoubleEditSlider * m_n2Value;
	QDoubleEditSlider * m_n3Value;
    QGroupBox * S1Grp;
	
	QDoubleEditSlider * m_a2Value;
	QDoubleEditSlider * m_b2Value;
	QDoubleEditSlider * m_m2Value;
	QDoubleEditSlider * m_n21Value;
	QDoubleEditSlider * m_n22Value;
	QDoubleEditSlider * m_n23Value;
    QGroupBox * S2Grp;
	
};

}
#endif