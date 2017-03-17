/*
 *  ActionIconFrame.h
 *  aphid
 *
 *  Created by jian zhang on 6/3/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <qt/QIconFrame.h>

namespace aphid {

class ActionIconFrame : public QIconFrame
{
Q_OBJECT

public:
    ActionIconFrame(QWidget *parent = 0);
	
	void setAction(int val);
	int getAction() const;
	
	virtual void mouseReleaseEvent(QMouseEvent *event);
	
signals:
	void actionTriggered(int a);

private:
	int m_action;
};

}