/*
 *  QDoubleEdit.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "QDoubleEdit.h"

namespace aphid {

QDoubleEdit::QDoubleEdit(const QModelIndex & idx, QWidget * parent) : QModelEdit(idx, parent)
{
	setValidator(&m_validate);
}

void QDoubleEdit::setValue(double x)
{
	m_value = x;
	QString t;
	t.setNum(m_value);
	setText(t);
}

double QDoubleEdit::value() 
{
	m_value = text().toDouble();
	return m_value;
}

}
