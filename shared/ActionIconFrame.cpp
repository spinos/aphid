/*
 *  ActionIconFrame.cpp
 *  knitfabric
 *
 *  Created by jian zhang on 6/3/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "ActionIconFrame.h"

ActionIconFrame::ActionIconFrame(QWidget *parent) : QIconFrame(parent) {}

void ActionIconFrame::setAction(int val)
{
	m_action = val;
}

int ActionIconFrame::getAction() const
{
	return m_action;
}

void ActionIconFrame::mouseReleaseEvent(QMouseEvent *event)
{
	QIconFrame::mouseReleaseEvent(event);
	emit actionTriggered(m_action);
}