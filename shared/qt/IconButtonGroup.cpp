/*
 *  IconButtonGroup.cpp
 *  
 *  push button with label, icon, name_id
 *
 *  Created by jian zhang on 9/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "IconButtonGroup.h"

namespace aphid {

IconButtonGroup::IconButtonGroup(const QIcon& icon,
		const QString & name, QWidget *parent)
    : QPushButton(icon, name, parent)
{
	connect(this, SIGNAL(pressed()),
            this, SLOT(sendPressedValue()));
		
}

void IconButtonGroup::sendPressedValue()
{
	QPair<int, int> val;
	val.first = m_nameId;
	val.second = 1;

	emit buttonPressed2(val);
}

void IconButtonGroup::setNameId(int x)
{ m_nameId = x; }

const int& IconButtonGroup::nameId() const
{ return m_nameId; }

}