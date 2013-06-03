/*
 *  untitled.cpp
 *  masq
 *
 *  Created by jian zhang on 5/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "ContextIconFrame.h"

ContextIconFrame::ContextIconFrame(QWidget *parent) : QIconFrame(parent) {}

void ContextIconFrame::setContext(int val)
{
	m_context = val;
}

int ContextIconFrame::getContext() const
{
	return m_context;
}

void ContextIconFrame::mousePressEvent(QMouseEvent *event)
{
	QIconFrame::mousePressEvent(event);
	if(getIconIndex() == 1) 
		emit contextEnabled(m_context);
}

void ContextIconFrame::mouseReleaseEvent(QMouseEvent *event)
{
}