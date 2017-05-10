/*
 *  StateIconFrame.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/29/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include <QtGui>
#include "StateIconFrame.h"

namespace aphid {

StateIconFrame::StateIconFrame(QWidget *parent) : QIconFrame(parent) {}

void StateIconFrame::mapState(int k, int x)
{
	m_stateMap[k] = x;
}

void StateIconFrame::setState(int val)
{
	std::map<int, int >::iterator it = m_stateMap.begin();
	for(;it!=m_stateMap.end();++it) {
		if(it->second == val) {
			setIconIndex(it->first);
			return;
		}
	}
}

int StateIconFrame::getState()
{
	return m_stateMap[getIconIndex()];
}

void StateIconFrame::mousePressEvent(QMouseEvent *event)
{
	QIconFrame::mousePressEvent(event);
	emit stateChanged(getState());
}

void StateIconFrame::mouseReleaseEvent(QMouseEvent *) {}

}