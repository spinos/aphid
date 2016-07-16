/*
 *  RedBlueControl.h
 *  cudafem
 *
 *  Created by jian zhang on Mon Jul 27 18:34:17 CST 2015 
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TTG_RED_BLUE_CONTROL_H
#define TTG_RED_BLUE_CONTROL_H

#include <QDialog>

QT_BEGIN_NAMESPACE
class QGroupBox;
class QCheckBox;
class QLabel;
QT_END_NAMESPACE

class QDoubleEditSlider;

namespace ttg {

class RedBlueControl : public QDialog
{
    Q_OBJECT

public:
    RedBlueControl(QWidget *parent = 0);
	
public slots:
	
private slots:
	void sendA(double x);
	void sendB(double x);
	void sendC(double x);
	void sendD(double x);
	
signals:
	void aChanged(double a);
	void bChanged(double a);
	void cChanged(double a);
	void dChanged(double a);
    
private:
    QDoubleEditSlider * m_aValue;
	QDoubleEditSlider * m_bValue;
	QDoubleEditSlider * m_cValue;
	QDoubleEditSlider * m_dValue;
	QGroupBox * S1Grp;
		
};

}
#endif