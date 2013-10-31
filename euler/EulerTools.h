/*
 *  EulerTools.h
 *  masq
 *
 *  Created by jian zhang on 5/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef EULERTOOLS_H
#define EULERTOOLS_H

#include <QToolBar>
#include <ToolContext.h>

class ContextIconFrame;
class ActionIconFrame;

class EulerTools : public QWidget, public ToolContext {
Q_OBJECT

public:	
	EulerTools(QWidget *parent = 0);
    ~EulerTools();
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

#endif