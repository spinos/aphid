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
class QLabel;
QT_END_NAMESPACE

class QIntEditSlider;
class QDoubleEditSlider;
class QSplineEdit;

class PhysicsControl : public QDialog
{
    Q_OBJECT

public:
    PhysicsControl(QWidget *parent = 0);
	
public slots:
	
private slots:
    void sendYoungModulus(double x);
    void sendStiffnessAttenuateEnds(QPointF v);
    void sendStiffnessAttenuateLeft(QPointF v);
    void sendStiffnessAttenuateRight(QPointF v);
signals:
	void youngsModulusChanged(double a);
    void stiffnessAttenuateEndsChanged(QPointF v);
    void stiffnessAttenuateLeftChanged(QPointF v);
    void stiffnessAttenuateRightChanged(QPointF v);
private:
    QGroupBox * YGrp;
    QDoubleEditSlider * m_youngModulusValue;
    QGroupBox * yAGrp;
    QLabel * stiffnessCurveLabel;
    QSplineEdit * m_youngAttenuateValue;
	
};
#endif