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
*/	
	createState();
	
	for(std::vector<StateIconFrame *>::iterator it = m_stateFrames.begin(); it != m_stateFrames.end(); ++it) {
		
	}
	
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
    ActionIconFrame * plantview = new ActionIconFrame(this);
	plantview->addIconFile(":/icons/plant.png");
	plantview->setIconIndex(0);
	plantview->setAction(gar::actViewPlant);
	
	ActionIconFrame * turfview = new ActionIconFrame(this);
	turfview->addIconFile(":/icons/turf.png");
	turfview->setIconIndex(0);
	turfview->setAction(gar::actViewTurf);
	
	m_actionFrames.push_back(plantview);
	m_actionFrames.push_back(turfview);
}

void ToolBox::createState()
{
	m_dspStateIcon = new StateIconFrame(this);
	m_dspStateIcon->addIconFile(":/icons/triangle.png");
	m_dspStateIcon->addIconFile(":/icons/dop.png");
	m_dspStateIcon->addIconFile(":/icons/point.png");
	m_dspStateIcon->addIconFile(":/icons/voxel.png");
	m_dspStateIcon->setIconIndex(0);
	m_dspStateIcon->mapState(0, gar::dsTriangle);
	m_dspStateIcon->mapState(1, gar::dsDop);
	m_dspStateIcon->mapState(2, gar::dsPoint);
	m_dspStateIcon->mapState(3, gar::dsVoxel);
	
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
