/*
 *  IntEditGroup.h
 *  
 *  [1,3] int field with label, name_id
 *
 *  Created by jian zhang on 9/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_INT_EDIT_GROUP_H
#define APH_INT_EDIT_GROUP_H

#include <QWidget>

QT_BEGIN_NAMESPACE
class QLabel;
class QLineEdit;
class QIntValidator;
class QPushButton;
QT_END_NAMESPACE

namespace aphid {

class IntEditGroup : public QWidget 
{
	Q_OBJECT
public:
	IntEditGroup(const QString & name, int numFields,
		QWidget *parent = 0);
	
	void setLimit(int bottom, int top);
	void setValue0(int x);
	int value0() const;
	
	void setValue1(int x);
	int value1() const;
	
	void setValue2(int x);
	int value2() const;
	
	void setValues(const int* v);
	void getValues(int* v) const;
	
	void setNameId(int x);
	const int& nameId() const;
	
private slots:
	void validateEditValue0();
	void validateEditValue1();
	void validateEditValue2();
	void spinValue0Down();
	void spinValue0Up();
	void spinValue1Down();
	void spinValue1Up();
	void spinValue2Down();
	void spinValue2Up();
	
signals:
	void valueChanged2(QPair<int, QVector<int> > x);
	
private:
	void sendValues();
	
	
private:
	QLabel * m_label;
	QLineEdit * m_edit[3];
	QPushButton* m_leftBtn[3];
	QPushButton* m_rightBtn[3];
	QIntValidator * m_validate;
	int m_numFields;
	int m_nameId;
	int m_bottomValue, m_topValue;
};

}
#endif