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
#include <StateIconFrame.h>

ToolBox::ToolBox(QWidget *parent) : QToolBar(parent) 
{
	createContext();
	
	createAction();
	
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
	
	for(std::vector<ActionIconFrame *>::iterator it = m_actionFrames.begin(); it != m_actionFrames.end(); ++it) {
		addWidget(*it);
		connect(*it, SIGNAL(actionTriggered(int)), this, SLOT(onActionFrameTriggered(int)));
	}
	
	setContext(SelectFace);
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
    ContextIconFrame * selectFace = new ContextIconFrame(this);
	selectFace->addIconFile(":selectFace.png");
	selectFace->addIconFile(":selectFaceActive.png");
	selectFace->setIconIndex(1);
	selectFace->setContext(SelectFace);
	
    ContextIconFrame * createContour = new ContextIconFrame(this);
	
	createContour->addIconFile(":createFeather.png");
	createContour->addIconFile(":createFeatherActive.png");
	createContour->setIconIndex(0);
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
	
	ContextIconFrame * lengthContour = new ContextIconFrame(this);
	
	lengthContour->addIconFile(":featherLength.png");
	lengthContour->addIconFile(":featherLengthActive.png");
	lengthContour->setIconIndex(0);
	lengthContour->setContext(ScaleBodyContourFeatherLength);
	
	ContextIconFrame * widthContour = new ContextIconFrame(this);
	
	widthContour->addIconFile(":featherWidth.png");
	widthContour->addIconFile(":featherWidthActive.png");
	widthContour->setIconIndex(0);
	widthContour->setContext(ScaleBodyContourFeatherWidth);
	
	ContextIconFrame * bendContour = new ContextIconFrame(this);
	
	bendContour->addIconFile(":roll.png");
	bendContour->addIconFile(":rollActive.png");
	bendContour->setIconIndex(0);
	bendContour->setContext(CurlBodyContourFeather);
	
	ContextIconFrame * moveLight = new ContextIconFrame(this);
	
	moveLight->addIconFile(":move.png");
	moveLight->addIconFile(":moveActive.png");
	moveLight->setIconIndex(0);
	moveLight->setContext(MoveTransform);
	
	ContextIconFrame * rotateLight = new ContextIconFrame(this);
	
	rotateLight->addIconFile(":rotate.png");
	rotateLight->addIconFile(":rotateActive.png");
	rotateLight->setIconIndex(0);
	rotateLight->setContext(RotateTransform);
	
	ContextIconFrame * paintMap = new ContextIconFrame(this);
	paintMap->addIconFile(":brush.png");
	paintMap->addIconFile(":brushActive.png");
	paintMap->setIconIndex(0);
	paintMap->setContext(PaintMap);
	
	ContextIconFrame * erect = new ContextIconFrame(this);
	erect->addIconFile(":pitch.png");
	erect->addIconFile(":pitchActive.png");
	erect->setIconIndex(0);
	erect->setContext(PitchBodyContourFeather);
	
	m_contextFrames.push_back(selectFace);
	m_contextFrames.push_back(paintMap);
	m_contextFrames.push_back(selectRegion);
	m_contextFrames.push_back(createContour);
	m_contextFrames.push_back(combContour);
	m_contextFrames.push_back(lengthContour);
	m_contextFrames.push_back(widthContour);
	m_contextFrames.push_back(bendContour);
	m_contextFrames.push_back(erect);
	m_contextFrames.push_back(eraseContour);
	m_contextFrames.push_back(moveLight);
	m_contextFrames.push_back(rotateLight);
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

void ToolBox::createState()
{
	StateIconFrame * toggleFeather = new StateIconFrame(this);
	toggleFeather->addIconFile(":eyeOpen.png");
	toggleFeather->addIconFile(":eyeClose.png");
	toggleFeather->setIconIndex(0);
	toggleFeather->setState(DisplayFeather);
	m_stateFrames.push_back(toggleFeather);
}

void ToolBox::onStateFrameChanged(int s)
{
	emit stateChanged(s);
}
