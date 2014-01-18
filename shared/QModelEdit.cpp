/*
 *  QModelEdit.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "QModelEdit.h"

QModelEdit::QModelEdit(const QModelIndex & idx, QWidget * parent) : QLineEdit(parent)
{
	m_index = idx;
}

QModelIndex QModelEdit::index() const
{
	return m_index;
}