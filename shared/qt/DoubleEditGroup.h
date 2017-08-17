/*
 *  DoubleEditGroup.h
 *
 *  [1,3] fields 
 *
 *  Created by jian zhang on 10/22/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_DOUBLE_EDIT_GROUP_H
#define APH_DOUBLE_EDIT_GROUP_H

#include <QGroupBox>

QT_BEGIN_NAMESPACE
class QLabel;
class QLineEdit;
class QDoubleValidator;
class QPushButton;
QT_END_NAMESPACE

class DoubleEditGroup : public QGroupBox 
{
	Q_OBJECT
public:
	DoubleEditGroup(const QString & name, int numFields, 
		QWidget *parent = 0);
	
	void setValues(const float* v);
	void getValues(float* v) const;
	void setNameId(int x);
	const int& nameId() const;
	
signals:
	void valueChanged2(QPair<int, QVector<double> > x);
	
private slots:
	void sendValues();
	void spinValue0Up();
	void spinValue0Down();
	void spinValue1Up();
	void spinValue1Down();
	void spinValue2Up();
	void spinValue2Down();
	
private:
	void spinValueDown(int i);
	void spinValueUp(int i);
	
private:
	QLabel * m_label;
	QLineEdit * m_edit[3];
	QPushButton* m_leftBtn[3];
	QPushButton* m_rightBtn[3];
	QDoubleValidator * m_validate;
	int m_numFields;
	int m_nameId;
};

#endif