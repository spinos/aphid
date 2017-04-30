/*
 *  toolBox.h
 *  garden
 *
 *  Created by jian zhang on 5/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef GAR_TOOLBOX_H
#define GAR_TOOLBOX_H

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
	void stateChanged(int s);
	
public slots:
	void onContextFrameChanged(int c);
	void onActionFrameTriggered(int a);
	void onStateFrameChanged(int s);
};

#endif