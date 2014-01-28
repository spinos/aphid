/*
 *  untitled.h
 *  masq
 *
 *  Created by jian zhang on 5/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <QIconFrame.h>

class ContextIconFrame : public QIconFrame
{
Q_OBJECT

public:
    ContextIconFrame(QWidget *parent = 0);
	
	void setContext(int val);
	int getContext() const;
protected:	
	virtual void mousePressEvent(QMouseEvent *event);
	virtual void mouseReleaseEvent(QMouseEvent *event);
signals:
	void contextEnabled(int c);
	void contextDisabled(int c);
private:
	int m_context;
};