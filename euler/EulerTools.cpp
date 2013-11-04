/*
 *  EulerTools.cpp
 *  masq
 *
 *  Created by jian zhang on 5/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "EulerTools.h"
#include <ContextIconFrame.h>
#include <ActionIconFrame.h>

EulerTools::EulerTools(QWidget *parent) : QWidget(parent) 
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
	setContext(CreateBodyContourFeather);
}

EulerTools::~EulerTools() {}

void EulerTools::onContextFrameChanged(int c)
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

void EulerTools::onActionFrameTriggered(int a)
{
	emit actionTriggered(a);
}

void EulerTools::createContext()
{
    ContextIconFrame * createContour = new ContextIconFrame(this);
	
	createContour->addIconFile(":move.png");
	createContour->addIconFile(":moveActive.png");
	createContour->setIconIndex(1);
	createContour->setContext(MoveTransform);
	
	ContextIconFrame * combContour = new ContextIconFrame(this);
	
	combContour->addIconFile(":rotate.png");
	combContour->addIconFile(":rotateActive.png");
	combContour->setIconIndex(0);
	combContour->setContext(RotateTransform);
	
	ContextIconFrame * eraseContour = new ContextIconFrame(this);
	
	eraseContour->addIconFile(":moveVertex.png");
	eraseContour->addIconFile(":moveVertexActive.png");
	eraseContour->setIconIndex(0);
	eraseContour->setContext(MoveMeshComponent);
	
	ContextIconFrame * smooth = new ContextIconFrame(this);
	
	smooth->addIconFile(":smoothSurface.png");
	smooth->addIconFile(":smoothSurfaceActive.png");
	smooth->setIconIndex(0);
	smooth->setContext(SmoothSurface);
	
	m_contextFrames.push_back(createContour);
	m_contextFrames.push_back(combContour);
	m_contextFrames.push_back(eraseContour);
	m_contextFrames.push_back(smooth);
}

void EulerTools::createAction()
{
    ActionIconFrame * rb = new ActionIconFrame(this);
	rb->addIconFile(":rebuild.png");
	rb->addIconFile(":rebuildActive.png");
	rb->setIconIndex(0);
	rb->setAction(RebuildBodyContourFeather);
	
	ActionIconFrame * clr = new ActionIconFrame(this);
	clr->addIconFile(":clear.png");
	clr->addIconFile(":clearActive.png");
	clr->setIconIndex(0);
	clr->setAction(ClearBodyContourFeather);
	
	ActionIconFrame * b = new ActionIconFrame(this);
	b->addIconFile(":bake.png");
	b->addIconFile(":bakeActive.png");
	b->setIconIndex(0);
	b->setAction(BakeAnimation);

	m_actionFrames.push_back(rb);
	m_actionFrames.push_back(clr);
	m_actionFrames.push_back(b);
}
