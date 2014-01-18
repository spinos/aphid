/*
 *  QIntEdit.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "QIntEdit.h"

QIntEdit::QIntEdit(const QModelIndex & idx, QWidget * parent) : QModelEdit(idx, parent)
{
	setValidator(&m_validate);
}

void QIntEdit::setValue(int x)
{
	m_value = x;
	QString t;
	t.setNum(m_value);
	setText(t);
}

int QIntEdit::value() 
{
	m_value = text().toInt();
	return m_value;
}
