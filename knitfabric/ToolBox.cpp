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
	
	ActionIconFrame * setWale = new ActionIconFrame(this);
	
	setWale->addIconFile(":setwale.png");
	setWale->addIconFile(":setwaleact.png");
	setWale->setIconIndex(0);
	setWale->setAction(1);
	
	m_actionFrames.push_back(setWale);
	
	for(std::vector<ContextIconFrame *>::iterator it = m_contextFrames.begin(); it != m_contextFrames.end(); ++it) {
		addWidget(*it);
		connect(*it, SIGNAL(contextEnabled(int)), this, SLOT(contextFrameChanged(int)));
	}
	
	addSeparator();
	
	for(std::vector<ActionIconFrame *>::iterator it = m_actionFrames.begin(); it != m_actionFrames.end(); ++it) {
		addWidget(*it);
		//connect(*it, SIGNAL(contextEnabled(int)), this, SLOT(contextFrameChanged(int)));
	}
}

ToolBox::~ToolBox() {}

void ToolBox::contextFrameChanged(int c)
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
