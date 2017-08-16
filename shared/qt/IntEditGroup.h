/*
 *  IntEditGroup.h
 *  
 *  int field with label, name_id
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
	IntEditGroup(const QString & name, QWidget *parent = 0);
	
	void setLimit(int bottom, int top);
	void setValue(int x);
	int value() const;
	void setNameId(int x);
	const int& nameId() const;
	
private slots:
	void setEditValue(int x);
	void validateEditValue();
	
signals:
	void valueChanged2(QPair<int, int> x);
	
private slots:
	void spinValueDown();
	void spinValueUp();
	
private:
	QLabel * m_label;
	QLineEdit * m_edit;
	QIntValidator * m_validate;
	QPushButton* m_leftBtn;
	QPushButton* m_rightBtn;
	int m_nameId;
	int m_bottomValue, m_topValue;
};

}
#endif