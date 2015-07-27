/*
 *  PhysicsControl.h
 *  cudafem
 *
 *  Created by jian zhang on Mon Jul 27 18:34:17 CST 2015 
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef PHYSICS_CONTROL_H
#define PHYSICS_CONTROL_H

#include <QDialog>

QT_BEGIN_NAMESPACE
class QGroupBox;
class QCheckBox;
QT_END_NAMESPACE

class QIntEditSlider;
class QDoubleEditSlider;

class PhysicsControl : public QDialog
{
    Q_OBJECT

public:
    PhysicsControl(QWidget *parent = 0);
	
public slots:
	
private slots:
    void sendYoungModulus(double x);
signals:
	void youngsModulusChanged(double a);
    
private:
    QGroupBox * YGrp;
    QDoubleEditSlider * m_youngModulusValue;
	
};
#endif