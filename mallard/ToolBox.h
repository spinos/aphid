/*
 *  ToolBox.h
 *  masq
 *
 *  Created by jian zhang on 5/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef TOOLBOX_H
#define TOOLBOX_H

#include <QToolBar>
#include <ToolContext.h>

class ContextIconFrame;
class ActionIconFrame;
class StateIconFrame;

class ToolBox : public QToolBar, public ToolContext {
Q_OBJECT

public:	
	ToolBox(QWidget *parent = 0);
    ~ToolBox();
private:
    void createContext();
    void createAction();
	void createState();
	std::vector<ContextIconFrame *> m_contextFrames;
	std::vector<ActionIconFrame *> m_actionFrames;
	std::vector<StateIconFrame *> m_stateFrames;
	
signals:
	void contextChanged(int c);
	void actionTriggered(int c);
	void stateChanged(int s);
	
public slots:
	void onContextFrameChanged(int c);
	void onActionFrameTriggered(int a);
	void onStateFrameChanged(int s);
};

#endif