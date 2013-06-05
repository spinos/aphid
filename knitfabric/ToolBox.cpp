/*
 *  ToolBox.cpp
 *  masq
 *
 *  Created by jian zhang on 5/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "ToolBox.h"
#include <ContextIconFrame.h>
#include <ActionIconFrame.h>

ToolBox::ToolBox(QWidget *parent) : QToolBar(parent) 
{
	createContext();
	
	createAction();
	
	for(std::vector<ContextIconFrame *>::iterator it = m_contextFrames.begin(); it != m_contextFrames.end(); ++it) {
		addWidget(*it);
		connect(*it, SIGNAL(contextEnabled(int)), this, SLOT(onContextFrameChanged(int)));
	}
	
	addSeparator();
	
	for(std::vector<ActionIconFrame *>::iterator it = m_actionFrames.begin(); it != m_actionFrames.end(); ++it) {
		addWidget(*it);
		connect(*it, SIGNAL(actionTriggered(int)), this, SLOT(onActionFrameTriggered(int)));
	}
}

ToolBox::~ToolBox() {}

void ToolBox::onContextFrameChanged(int c)
{
	setContext((InteractMode)c);
	for(std::vector<ContextIconFrame *>::iterator it = m_contextFrames.begin(); it != m_contextFrames.end(); ++it) {
		if((*it)->getContext() != c)
			(*it)->setIconIndex(0);
		else
			(*it)->setIconIndex(1);
	}
	emit contextChanged(c);
}

void ToolBox::onActionFrameTriggered(int a)
{
	emit actionTriggered(a);
}

void ToolBox::createContext()
{
    ContextIconFrame * selectComponent = new ContextIconFrame(this);
	
	selectComponent->addIconFile(":selvertex.png");
	selectComponent->addIconFile(":selvertexact.png");
	selectComponent->setIconIndex(1);
	selectComponent->setContext(SelectVertex);
	
	ContextIconFrame * selectAnchor = new ContextIconFrame(this);
	selectAnchor->addIconFile(":seledge.png");
	selectAnchor->addIconFile(":seledgeact.png");
	selectAnchor->setIconIndex(0);
	selectAnchor->setContext(SelectEdge);
	
	m_contextFrames.push_back(selectComponent);
	m_contextFrames.push_back(selectAnchor);
}

void ToolBox::createAction()
{
    ActionIconFrame * setWale = new ActionIconFrame(this);
	
	setWale->addIconFile(":setwale.png");
	setWale->addIconFile(":setwaleact.png");
	setWale->setIconIndex(0);
	setWale->setAction(SetWaleEdge);
	
	ActionIconFrame * increaseWale = new ActionIconFrame(this);
	increaseWale->addIconFile(":increasewale.png");
	increaseWale->addIconFile(":increasewaleact.png");
	increaseWale->setIconIndex(0);
	increaseWale->setAction(SetWaleEdge);
	
	ActionIconFrame * decreaseWale = new ActionIconFrame(this);
	decreaseWale->addIconFile(":decreasewale.png");
	decreaseWale->addIconFile(":decreasewaleact.png");
	decreaseWale->setIconIndex(0);
	decreaseWale->setAction(SetWaleEdge);

	m_actionFrames.push_back(setWale);
	m_actionFrames.push_back(increaseWale);
	m_actionFrames.push_back(decreaseWale);
}
