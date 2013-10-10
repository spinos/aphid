/*
 *  BaseScene.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/26/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseScene.h"

BaseScene::BaseScene() : BaseFile("untitled")
{
	m_isDirty = false;
}

BaseScene::~BaseScene() {}

void BaseScene::setDirty()
{
	m_isDirty = true;
}

void BaseScene::setClean()
{
	m_isDirty = false;
}

bool BaseScene::isDirty() const
{
	return m_isDirty;
}

bool BaseScene::newScene()
{
	if(isDirty())
		if(!discardConfirm()) return false;
	
	clearScene();
	
	return true;
}

bool BaseScene::openScene(const std::string & fileName)
{
	if(isDirty()) {
		if(!discardConfirm()) return false;
	}
	
	clearScene();
	
	if(!FileExists(fileName)) {
		setLatestError(BaseFile::FileNotFound);
		return false;
	}

	if(!readSceneFromFile(fileName)) return false;

	setClean();
	
	return open(fileName);
}

bool BaseScene::saveScene()
{
	if(writeSceneToFile(fileName())) {
		setClean();
		return true;
	}
	return false;
}

bool BaseScene::saveSceneAs(const std::string & fileName)
{
	setFileName(fileName);
	return saveScene();
}

bool BaseScene::revertScene()
{
	std::string fileToRevert = fileName();
	clearScene();
	readSceneFromFile(fileToRevert);
	setFileName(fileToRevert);
	setClean();
	return true;
}

bool BaseScene::shouldSave()
{
	return false;
}

bool BaseScene::discardConfirm()
{
	return true;
}

void BaseScene::clearScene()
{
	setClean();
	setFileName("untitled");
}

bool BaseScene::writeSceneToFile(const std::string & fileName)
{
	return true;
}

bool BaseScene::readSceneFromFile(const std::string & fileName)
{
	return true;
}
