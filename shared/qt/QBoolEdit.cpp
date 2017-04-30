/*
 *  QBoolEdit.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "QBoolEdit.h"

QBoolEdit::QBoolEdit(const QModelIndex & idx, QWidget * parent) : QModelEdit(idx, parent)
{
}

void QBoolEdit::setValue(bool x)
{
	m_value = x;
	setText(translateBoolToStr(x));
}

bool QBoolEdit::value() 
{
	m_value = translateStrToBool(text());
	return m_value;
}

QString QBoolEdit::translateBoolToStr(bool src) const
{
	if(src) return tr("true");
	return tr("false");
}

bool QBoolEdit::translateStrToBool(const QString & src) const
{
	if(src == "on" || src == "1" || src == "true") return true;
	return false;
}
