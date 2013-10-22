/*
 *  QDouble3Edit.h
 *  eulerRot
 *
 *  Created by jian zhang on 10/22/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
#include <QWidget>

QT_BEGIN_NAMESPACE
class QLabel;
class QLineEdit;
class QDoubleValidator;
QT_END_NAMESPACE

class QDouble3Edit : public QWidget 
{
	Q_OBJECT
public:
	QDouble3Edit(const QString & name, QWidget *parent = 0);
	
	void setValue(const Vector3F & v);
	Vector3F value() const;
	
	void setDOF(const Float3 & dof);
	
private slots:
	void validateEditValue();
	
signals:
	void valueChanged(Vector3F v);
	
private:

private:
	QLabel * m_label;
	QLineEdit * m_edit0;
	QLineEdit * m_edit1;
	QLineEdit * m_edit2;
	QDoubleValidator * m_validate;
};