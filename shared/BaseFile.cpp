/*
 *  BaseFile.cpp
 *  mallard
 *
 *  Created by jian zhang on 10/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseFile.h"
#include <iostream>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/convenience.hpp>
BaseFile::BaseFile() 
{
	m_fileName = "unknown";
	_opened = 0;
	m_dirty = 0;
	m_clearMode = Normal;
	m_error = NoError;
}

BaseFile::BaseFile(const char * name)
{
	m_fileName = std::string(name);
	_opened = 0;
	m_dirty = 0;
	m_clearMode = Normal;
	m_error = NoError;
}

BaseFile::~BaseFile() 
{ close(); }

bool BaseFile::clear()
{
	if(shouldSave())
		if(!confirmDiscardChanges()) return false;
	
	doClear();
	
	return true;
}

bool BaseFile::create(const std::string & filename)
{
	if(!doCreate(filename)) return false;
	
	setFileName(filename);
	setOpened();
	return true;
}

bool BaseFile::open()
{
	std::string s = chooseOpenFileName();
	if(s == "") return false;
	m_clearMode = Normal;
	return open(s);
}

bool BaseFile::open(const std::string & fileName)
{
	if(!FileExists(fileName)) {
		setLatestError(BaseFile::FileNotFound);
		return false;
	}
	
	if(!clear()) return false;
	
	doClose();
	
	if(!doRead(fileName)) return false;
	
	afterOpen();
	setFileName(fileName);
	setOpened();
	return true;
}

bool BaseFile::save()
{
	if(!shouldSave()) return false;
	if(isUntitled()) {
		std::string s = chooseSaveFileName();
		if(s == "") return false;
		return saveAs(s);
	}
	beforeSave();
	if(!doWrite(fileName())) return false;
	setClean();
    return true;
}

bool BaseFile::saveAs(const std::string & name)
{
	std::string s = name;
	if(s == "") {
		s = chooseSaveFileName();
		if(s == "") return false;
	}
	beforeSave();
	if(!doWrite(s)) return false;
	setFileName(s);
	setClean();
    return true;
}

bool BaseFile::revert()
{
	if(isUntitled()) return false;
	m_clearMode = Revert;
	return open(fileName());
}

bool BaseFile::close()
{
	m_clearMode = Close;
	if(!clear()) return false;
	doClose();
    return true;
}

bool BaseFile::shouldSave()
{
	return isDirty();
}

bool BaseFile::confirmDiscardChanges()
{
	return true;
}

std::string BaseFile::chooseOpenFileName()
{
	return "";
}

std::string BaseFile::chooseSaveFileName()
{
	return "";
}

void BaseFile::doClear()
{
	setClean();
	// setFileName("untitled");
}

bool BaseFile::doCreate(const std::string & fileName)
{
	return true;
}

bool BaseFile::doRead(const std::string & fileName)
{
	std::cout<<"\n base file do read "<<fileName;
	return true;
}

bool BaseFile::doWrite(const std::string & fileName)
{
	return true;
}

void BaseFile::doClose() 
{
	setClosed();
}

void BaseFile::setFileName(const std::string & filename)
{
    m_fileName = filename;
}

std::string BaseFile::fileName() const
{
	return m_fileName;
}

void BaseFile::setOpened()
{
    _opened = true;
}

void BaseFile::setClosed()
{
    _opened = false;
}

bool BaseFile::isOpened() const
{
	return _opened;
}

bool BaseFile::isUntitled() const
{
	return m_fileName == "untitled";
}

bool BaseFile::FileExists(const std::string & name)
{
	if(!boost::filesystem::exists(name)) {
		std::cout<<"WARNING: file "<<name<<" doesn't exist!";
		return false;
	}
	return true;
}

bool BaseFile::InvalidFilename(const std::string & name)
{
	return (name.size() < 3 || name == "unknown");
}

void BaseFile::setLatestError(BaseFile::ErrorMsg err)
{
	m_error = err;
}

BaseFile::ErrorMsg BaseFile::latestError() const
{
	return m_error;
}

void BaseFile::setDirty()
{
	m_dirty = true;
}

void BaseFile::setClean()
{
	m_dirty = false;
}

bool BaseFile::isDirty() const
{
	return m_dirty;
}

bool BaseFile::isReverting() const
{
	return m_clearMode == Revert;
}

bool BaseFile::isClosing() const
{
	return m_clearMode == Close;
}

void BaseFile::beforeSave() {}

void BaseFile::afterOpen() {}

bool BaseFile::doCopy(const std::string & filename) 
{
	return true;
}
