/*
 *  BaseScene.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/26/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseScene.h"
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/convenience.hpp>

BaseScene::BaseScene() : m_isDirty(false) 
{
	m_fileName = "untitled";
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

bool BaseScene::isUntitled() const
{
	return m_fileName == "untitled";
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
		clearScene();
	}
	
	if(!fileExists(fileName)) {
		m_error = FileNotFound;
		return false;
	}
	
	readSceneFromFile(fileName);
	m_fileName = fileName;
	setClean();
	return true;
}

bool BaseScene::saveScene()
{
	if(writeSceneToFile(m_fileName)) {
		setClean();
		return true;
	}
	return false;
}

bool BaseScene::saveSceneAs(const std::string & fileName)
{
	m_fileName = fileName;
	return saveScene();
}

bool BaseScene::fileExists(const std::string & fileName)
{
	return boost::filesystem::exists(fileName);
}

void BaseScene::setLatestError(BaseScene::ErrorMsg err)
{
	m_error = err;
}

BaseScene::ErrorMsg BaseScene::latestError() const
{
	return m_error;
}

bool BaseScene::discardConfirm()
{
	return true;
}

void BaseScene::clearScene()
{
	setClean();
}

bool BaseScene::writeSceneToFile(const std::string & fileName)
{
	return true;
}

bool BaseScene::readSceneFromFile(const std::string & fileName)
{
	return true;
}
