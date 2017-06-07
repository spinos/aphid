/*
 *  toolBox.cpp
 *  hes viewer
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

using namespace aphid;

ToolBox::ToolBox(QWidget *parent) : QToolBar(parent) 
{
	createAction();
	for(std::vector<ActionIconFrame *>::iterator it = m_actionFrames.begin(); it != m_actionFrames.end(); ++it) {
		addWidget(*it);
		connect(*it, SIGNAL(actionTriggered(int)), this, SLOT(onActionFrameTriggered(int)));
	}	
	createState();
	
	for(std::vector<StateIconFrame *>::iterator it = m_stateFrames.begin(); it != m_stateFrames.end(); ++it) {
		
	}
	
}

ToolBox::~ToolBox() 
{}

void ToolBox::onContextFrameChanged(int c)
{
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
}

void ToolBox::createState()
{
	m_dspStateIcon = new StateIconFrame(this);
	m_dspStateIcon->addIconFile(":/icons/triangle.png");
	m_dspStateIcon->setIconIndex(0);
	m_dspStateIcon->mapState(0, 0);
	
	connect(m_dspStateIcon, SIGNAL(stateChanged(int)), this, SLOT(onDspStateChanged(int)));
	
	addWidget(m_dspStateIcon);
}

void ToolBox::onDspStateChanged(int s)
{
	emit dspStateChanged(s);
}

void ToolBox::setDisplayState(int x)
{
	m_dspStateIcon->setState(x);
}
