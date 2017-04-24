/*
 *  toolBox.cpp
 *  garden
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
#include "gar_common.h"

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
    ActionIconFrame * graphview = new ActionIconFrame(this);
	graphview->addIconFile(":/icons/graph.png");
	graphview->setIconIndex(0);
	graphview->setAction(gar::actViewGraph);
	
    ActionIconFrame * plantview = new ActionIconFrame(this);
	plantview->addIconFile(":/icons/plant.png");
	plantview->setIconIndex(0);
	plantview->setAction(gar::actViewPlant);
	
	ActionIconFrame * turfview = new ActionIconFrame(this);
	turfview->addIconFile(":/icons/turf.png");
	turfview->setIconIndex(0);
	turfview->setAction(gar::actViewTurf);
	
	m_actionFrames.push_back(graphview);
	m_actionFrames.push_back(plantview);
	m_actionFrames.push_back(turfview);
}

void ToolBox::createState()
{
}

void ToolBox::onStateFrameChanged(int s)
{
	emit stateChanged(s);
}
