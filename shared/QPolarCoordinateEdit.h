/*
 *  QPolarCoordinateEdit.h
 *  
 *
 *  Created by jian zhang on 7/30/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef Q_POLAR_COORDINATE_EDIT_H
#define Q_POLAR_COORDINATE_EDIT_H

#include <QWidget>

QT_BEGIN_NAMESPACE
class QLabel;
QT_END_NAMESPACE
class QAngleEdit;
class QPolarCoordinateEdit : public QWidget 
{
	Q_OBJECT
public:
	QPolarCoordinateEdit(const QString & name, QWidget *parent = 0);
	
	void setPhi(double x);
	void setTheta(double x);
	
private slots:
	void sendPhi(double x);
	void sendTheta(double x);
protected:
	
signals:
	void valueChanged(QPointF a);
	
private:
	
private:
	QLabel * m_name;
	QAngleEdit * m_theta;
	QAngleEdit * m_phi;
};
#endif