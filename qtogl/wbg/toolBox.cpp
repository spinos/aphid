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
#include <iostream>

using namespace aphid;

ToolBox::ToolBox(QWidget *parent) : QToolBar(parent) 
{
	createContext();
	
	for(std::vector<ContextIconFrame *>::iterator it = m_contextFrames.begin(); it != m_contextFrames.end(); ++it) {
		addWidget(*it);
		connect(*it, SIGNAL(contextEnabled(int)), this, SLOT(onContextFrameChanged(int)));
		connect(*it, SIGNAL(contextDisabled(int)), this, SLOT(onContextFrameChanged(int)));
	}
	//addSeparator();
/*
	
	createAction();
	
	
	createState();
	
	for(std::vector<StateIconFrame *>::iterator it = m_stateFrames.begin(); it != m_stateFrames.end(); ++it) {
		addWidget(*it);
		connect(*it, SIGNAL(stateChanged(int)), this, SLOT(onStateFrameChanged(int)));
	}
	
	for(std::vector<ActionIconFrame *>::iterator it = m_actionFrames.begin(); it != m_actionFrames.end(); ++it) {
		addWidget(*it);
		connect(*it, SIGNAL(actionTriggered(int)), this, SLOT(onActionFrameTriggered(int)));
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
    ContextIconFrame * selectFace = new ContextIconFrame(this);
	selectFace->addIconFile(":/icons/perspView.png");
	selectFace->addIconFile(":/icons/perspView.png");
	selectFace->setIconIndex(1);
	//selectFace->setContext(SelectFace);
	
    ContextIconFrame * createContour = new ContextIconFrame(this);
	createContour->addIconFile(":/icons/topView.png");
	createContour->addIconFile(":/icons/topView.png");
	createContour->setIconIndex(0);
	//createContour->setContext(CreateBodyContourFeather);
	
	m_contextFrames.push_back(selectFace);
	m_contextFrames.push_back(createContour);
	
}

void ToolBox::createAction()
{
    ActionIconFrame * rb = new ActionIconFrame(this);
	rb->addIconFile(":/icons/perspView.png");
	rb->addIconFile(":/icons/perspView.png");
	rb->setIconIndex(0);
	//rb->setAction(RebuildBodyContourFeather);

	m_actionFrames.push_back(rb);
	
}

void ToolBox::createState()
{
	StateIconFrame * toggleFeather = new StateIconFrame(this);
	toggleFeather->addIconFile(":/icons/topView.png");
	toggleFeather->addIconFile(":/icons/topView.png");
	toggleFeather->setIconIndex(0);
	toggleFeather->setState(0);
	m_stateFrames.push_back(toggleFeather);
}

void ToolBox::onStateFrameChanged(int s)
{
	emit stateChanged(s);
}
