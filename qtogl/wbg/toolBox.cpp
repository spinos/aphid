/*
 *  toolBox.cpp
 *  wbg
 *
 *  Created by jian zhang on 5/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "toolBox.h"
#include <qt/ContextIconFrame.h>
#include <qt/ActionIconFrame.h>
#include <qt/StateIconFrame.h>
#include "wbg_common.h"

using namespace aphid;

ToolBox::ToolBox(QWidget *parent) : QToolBar(parent) 
{
	createAction();
	for(std::vector<ActionIconFrame *>::iterator it = m_actionFrames.begin(); it != m_actionFrames.end(); ++it) {
		addWidget(*it);
		connect(*it, SIGNAL(actionTriggered(int)), this, SLOT(onActionFrameTriggered(int)));
	}
/*	
	createContext();
	
	for(std::vector<ContextIconFrame *>::iterator it = m_contextFrames.begin(); it != m_contextFrames.end(); ++it) {
		addWidget(*it);
		connect(*it, SIGNAL(contextEnabled(int)), this, SLOT(onContextFrameChanged(int)));
		connect(*it, SIGNAL(contextDisabled(int)), this, SLOT(onContextFrameChanged(int)));
	}
	//addSeparator();
	
	createState();
	
	for(std::vector<StateIconFrame *>::iterator it = m_stateFrames.begin(); it != m_stateFrames.end(); ++it) {
		addWidget(*it);
		connect(*it, SIGNAL(stateChanged(int)), this, SLOT(onStateFrameChanged(int)));
	}
*/	
}

ToolBox::~ToolBox() 
{}

void ToolBox::onContextFrameChanged(int c)
{
/*
	InteractMode cur = getContext();
	if(cur == (InteractMode)c) {
		for(std::vector<ContextIconFrame *>::iterator it = m_contextFrames.begin(); it != m_contextFrames.end(); ++it) {
			if((*it)->getContext() == c)
				(*it)->setIconIndex(1);
		}
		return;
	}
	
	setPreviousContext(cur);
	setContext((InteractMode)c);
	
	for(std::vector<ContextIconFrame *>::iterator it = m_contextFrames.begin(); it != m_contextFrames.end(); ++it) {
		if((*it)->getContext() != c)
			(*it)->setIconIndex(0);
		else
			(*it)->setIconIndex(1);
	}*/
	emit contextChanged(c);
}

void ToolBox::onActionFrameTriggered(int a)
{
	emit actionTriggered(a);
}

void ToolBox::createContext()
{	
}

void ToolBox::createAction()
{
    ActionIconFrame * topview = new ActionIconFrame(this);
	topview->addIconFile(":/icons/topView.png");
	topview->addIconFile(":/icons/topView.png");
	topview->setIconIndex(0);
	topview->setAction(wbg::actViewTop);
	
    ActionIconFrame * perspview = new ActionIconFrame(this);
	perspview->addIconFile(":/icons/perspView.png");
	perspview->addIconFile(":/icons/perspView.png");
	perspview->setIconIndex(0);
	perspview->setAction(wbg::actViewPersp);
	
	m_actionFrames.push_back(topview);
	m_actionFrames.push_back(perspview);
	
}

void ToolBox::createState()
{
}

void ToolBox::onStateFrameChanged(int s)
{
	emit stateChanged(s);
}
