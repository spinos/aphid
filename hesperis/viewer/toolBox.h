/*
 *  toolBox.h
 *  hes viewer
 *
 *  Created by jian zhang on 5/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef HES_TOOLBOX_H
#define HES_TOOLBOX_H

#include <QToolBar>

namespace aphid {

class ContextIconFrame;
class ActionIconFrame;
class StateIconFrame;

}

class ToolBox : public QToolBar {
Q_OBJECT

public:	
	ToolBox(QWidget *parent = 0);
    ~ToolBox();
	
	void setDisplayState(int x);
	
private:
    void createContext();
    void createAction();
	void createState();
	std::vector<aphid::ContextIconFrame *> m_contextFrames;
	std::vector<aphid::ActionIconFrame *> m_actionFrames;
	std::vector<aphid::StateIconFrame *> m_stateFrames;
	
signals:
	void contextChanged(int c);
	void actionTriggered(int c);
	void dspStateChanged(int s);
	
public slots:
	void onContextFrameChanged(int c);
	void onActionFrameTriggered(int a);
	void onDspStateChanged(int s);

private:
	aphid::StateIconFrame * m_dspStateIcon;
	
};

#endif