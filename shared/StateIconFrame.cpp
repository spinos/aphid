/*
 *  StateIconFrame.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/29/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "StateIconFrame.h"

StateIconFrame::StateIconFrame(QWidget *parent) : QIconFrame(parent) {}

void StateIconFrame::setState(int val)
{
	m_state = val;
}

int StateIconFrame::getState() const
{
	return m_state;
}

void StateIconFrame::mousePressEvent(QMouseEvent *event)
{
	QIconFrame::mousePressEvent(event);
	emit stateChanged(m_state);
}

void StateIconFrame::mouseReleaseEvent(QMouseEvent *) {}