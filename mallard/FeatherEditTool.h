/*
 *  FeatherEditTool.h
 *  mallard
 *
 *  Created by jian zhang on 10/3/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <QWidget>
#include <ToolContext.h>

class ContextIconFrame;
class ActionIconFrame;

class FeatherEditTool : public QWidget, public ToolContext {
Q_OBJECT

public:	
	FeatherEditTool(QWidget *parent = 0);
    ~FeatherEditTool();
private:
    void createContext();
    void createAction();
	std::vector<ContextIconFrame *> m_contextFrames;
	std::vector<ActionIconFrame *> m_actionFrames;
	
signals:
	void contextChanged(int c);
	void actionTriggered(int c);
	
public slots:
	void onContextFrameChanged(int c);
	void onActionFrameTriggered(int a);
};