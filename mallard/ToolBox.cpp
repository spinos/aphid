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
		connect(*it, SIGNAL(contextDisabled(int)), this, SLOT(onContextFrameChanged(int)));
	}
	
	addSeparator();
	
	for(std::vector<ActionIconFrame *>::iterator it = m_actionFrames.begin(); it != m_actionFrames.end(); ++it) {
		addWidget(*it);
		connect(*it, SIGNAL(actionTriggered(int)), this, SLOT(onActionFrameTriggered(int)));
	}
	
	setContext(CreateBodyContourFeather);
}

ToolBox::~ToolBox() {}

void ToolBox::onContextFrameChanged(int c)
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

void ToolBox::onActionFrameTriggered(int a)
{
	emit actionTriggered(a);
}

void ToolBox::createContext()
{
    
    ContextIconFrame * createContour = new ContextIconFrame(this);
	
	createContour->addIconFile(":brush.png");
	createContour->addIconFile(":brushActive.png");
	createContour->setIconIndex(1);
	createContour->setContext(CreateBodyContourFeather);
	
	ContextIconFrame * selectRegion = new ContextIconFrame(this);
	
	selectRegion->addIconFile(":colorPicker.png");
	selectRegion->addIconFile(":colorPickerActive.png");
	selectRegion->setIconIndex(0);
	selectRegion->setContext(SelectByColor);
	
	ContextIconFrame * combContour = new ContextIconFrame(this);
	
	combContour->addIconFile(":comb.png");
	combContour->addIconFile(":combActive.png");
	combContour->setIconIndex(0);
	combContour->setContext(CombBodyContourFeather);
	
	ContextIconFrame * eraseContour = new ContextIconFrame(this);
	
	eraseContour->addIconFile(":eraser.png");
	eraseContour->addIconFile(":eraserActive.png");
	eraseContour->setIconIndex(0);
	eraseContour->setContext(EraseBodyContourFeather);
	
	ContextIconFrame * scaleContour = new ContextIconFrame(this);
	
	scaleContour->addIconFile(":ruler.png");
	scaleContour->addIconFile(":rulerActive.png");
	scaleContour->setIconIndex(0);
	scaleContour->setContext(ScaleBodyContourFeather);
	
	ContextIconFrame * bendContour = new ContextIconFrame(this);
	
	bendContour->addIconFile(":pitch.png");
	bendContour->addIconFile(":pitchActive.png");
	bendContour->setIconIndex(0);
	bendContour->setContext(PitchBodyContourFeather);
	
	ContextIconFrame * smoothContour = new ContextIconFrame(this);
	
	smoothContour->addIconFile(":deintersectInactive.png");
	smoothContour->addIconFile(":deintersectActive.png");
	smoothContour->setIconIndex(0);
	smoothContour->setContext(Deintersect);
	
	m_contextFrames.push_back(createContour);
	m_contextFrames.push_back(selectRegion);
	m_contextFrames.push_back(combContour);
	m_contextFrames.push_back(scaleContour);
	m_contextFrames.push_back(bendContour);
	m_contextFrames.push_back(eraseContour);
	m_contextFrames.push_back(smoothContour);
}

void ToolBox::createAction()
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
	
	ActionIconFrame * render = new ActionIconFrame(this);
	render->addIconFile(":render.png");
	render->addIconFile(":renderActive.png");
	render->setIconIndex(0);
	render->setAction(LaunchRender);

	m_actionFrames.push_back(rb);
	m_actionFrames.push_back(clr);
	m_actionFrames.push_back(b);
	m_actionFrames.push_back(render);
}
