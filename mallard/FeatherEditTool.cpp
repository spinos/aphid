/*
 *  FeatherEditTool.cpp
 *  mallard
 *
 *  Created by jian zhang on 10/3/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "FeatherEditTool.h"
#include <QtGui>
#include <ContextIconFrame.h>
#include <ActionIconFrame.h>

FeatherEditTool::FeatherEditTool(QWidget *parent) : QWidget(parent) 
{
	createContext();
	
	createAction();
	
	QHBoxLayout * layout = new QHBoxLayout;
	
	for(std::vector<ContextIconFrame *>::iterator it = m_contextFrames.begin(); it != m_contextFrames.end(); ++it) {
		layout->addWidget(*it);
		connect(*it, SIGNAL(contextEnabled(int)), this, SLOT(onContextFrameChanged(int)));
		connect(*it, SIGNAL(contextDisabled(int)), this, SLOT(onContextFrameChanged(int)));
	}
	
	layout->addSpacing(32);
	
	for(std::vector<ActionIconFrame *>::iterator it = m_actionFrames.begin(); it != m_actionFrames.end(); ++it) {
		layout->addWidget(*it);
		connect(*it, SIGNAL(actionTriggered(int)), this, SLOT(onActionFrameTriggered(int)));
	}
	
	layout->addStretch();
	
	setLayout(layout);
	setContentsMargins(0, 0, 0, 0);
	layout->setContentsMargins(2, 2, 2, 2);
	
	setContext(MoveInUV);
}

FeatherEditTool::~FeatherEditTool() {}

void FeatherEditTool::onContextFrameChanged(int c)
{
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
	}
	emit contextChanged(c);
}

void FeatherEditTool::onActionFrameTriggered(int a)
{
	emit actionTriggered(a);
}

void FeatherEditTool::createContext()
{
    ContextIconFrame * move = new ContextIconFrame(this);
	
	move->addIconFile(":move.png");
	move->addIconFile(":moveActive.png");
	move->setIconIndex(1);
	move->setContext(MoveInUV);
	
	ContextIconFrame * mv = new ContextIconFrame(this);
	
	mv->addIconFile(":moveVertex.png");
	mv->addIconFile(":moveVertexActive.png");
	mv->setIconIndex(0);
	mv->setContext(MoveVertexInUV);
	
	m_contextFrames.push_back(move);
	m_contextFrames.push_back(mv);
}

void FeatherEditTool::createAction()
{
    ActionIconFrame * addFeather = new ActionIconFrame(this);
	addFeather->addIconFile(":addFeather.png");
	addFeather->addIconFile(":addFeatherActive.png");
	addFeather->setIconIndex(0);
	addFeather->setAction(AddFeatherExample);
	
	ActionIconFrame * rmFeather = new ActionIconFrame(this);
	rmFeather->addIconFile(":deleteFeather.png");
	rmFeather->addIconFile(":deleteFeatherActive.png");
	rmFeather->setIconIndex(0);
	rmFeather->setAction(RemoveFeatherExample);
	
	ActionIconFrame * increaseWale = new ActionIconFrame(this);
	increaseWale->addIconFile(":increasewale.png");
	increaseWale->addIconFile(":increasewaleact.png");
	increaseWale->setIconIndex(0);
	increaseWale->setAction(IncreaseWale);
	
	ActionIconFrame * decreaseWale = new ActionIconFrame(this);
	decreaseWale->addIconFile(":decreasewale.png");
	decreaseWale->addIconFile(":decreasewaleact.png");
	decreaseWale->setIconIndex(0);
	decreaseWale->setAction(DecreaseWale);
	
	ActionIconFrame * increaseCourse = new ActionIconFrame(this);
	increaseCourse->addIconFile(":increasecourse.png");
	increaseCourse->addIconFile(":increasecourseact.png");
	increaseCourse->setIconIndex(0);
	increaseCourse->setAction(IncreaseCourse);
	
	ActionIconFrame * decreaseCourse = new ActionIconFrame(this);
	decreaseCourse->addIconFile(":decreasecourse.png");
	decreaseCourse->addIconFile(":decreasecourseact.png");
	decreaseCourse->setIconIndex(0);
	decreaseCourse->setAction(DecreaseCourse);

	m_actionFrames.push_back(addFeather);
	m_actionFrames.push_back(rmFeather);
	m_actionFrames.push_back(increaseWale);
	m_actionFrames.push_back(decreaseWale);
	m_actionFrames.push_back(increaseCourse);
	m_actionFrames.push_back(decreaseCourse);
}