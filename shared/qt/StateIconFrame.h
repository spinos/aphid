/*
 *  StateIconFrame.h
 *  aphid
 *
 *  Created by jian zhang on 1/29/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include <map>
#include <qt/QIconFrame.h>

namespace aphid {

class StateIconFrame : public QIconFrame
{
Q_OBJECT

public:
    StateIconFrame(QWidget *parent = 0);
	
	void mapState(int k, int x);
	void setState(int val);
	int getState();
	
protected:	
	virtual void mousePressEvent(QMouseEvent *event);
	virtual void mouseReleaseEvent(QMouseEvent *event);
signals:
	void stateChanged(int a);

private:
	std::map<int, int > m_stateMap;
};

}