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
class QDouble3Edit;
class QSplineEdit;
class QPolarCoordinateEdit;

class PhysicsControl : public QDialog
{
    Q_OBJECT

public:
    PhysicsControl(QWidget *parent = 0);
	
public slots:
	
private slots:
	void sendDensity(double x);
    void sendYoungModulus(double x);
    void sendStiffnessAttenuateEnds(QPointF v);
    void sendStiffnessAttenuateLeft(QPointF v);
    void sendStiffnessAttenuateRight(QPointF v);
	void sendWindSpeed(double x);
	void sendWindVec(QPointF v);
	void sendWindTurbulence(double x);
	void sendGravity(Vector3F v);
signals:
	void densityChanged(double x);
	void youngsModulusChanged(double a);
    void stiffnessAttenuateEndsChanged(QPointF v);
    void stiffnessAttenuateLeftChanged(QPointF v);
    void stiffnessAttenuateRightChanged(QPointF v);
	void windVecChanged(QPointF v);
	void windSpeedChanged(double x);
	void windTurbulenceChanged(double x);
	void gravityChanged(Vector3F v);
private:
    QGroupBox * YGrp;
    QDoubleEditSlider * m_youngModulusValue;
    QGroupBox * yAGrp;
    QLabel * stiffnessCurveLabel;
    QSplineEdit * m_youngAttenuateValue;
	QGroupBox * dsGrp;
	QDoubleEditSlider * m_densityValue;
	QDoubleEditSlider * m_windSpeedValue;
	QPolarCoordinateEdit * m_windVecValue;
	QDoubleEditSlider * m_windTurbulenceValue;
	QDouble3Edit * m_gravityValue;
};
#endif