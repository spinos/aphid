/*
 *  BaseFile.h
 *  mallard
 *
 *  Created by jian zhang on 10/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <string>

class BaseFile {
public:
	enum ErrorMsg {
		NoError = 0,
		FileNotFound = 1,
		FileNotWritable = 2,
		FileNotReadable = 3
	};
	
	BaseFile();
	BaseFile(const char * name);
	virtual ~BaseFile();
	
	bool clear();
	bool create(const std::string & filename);
	bool open();
	bool open(const std::string & filename);
	bool save();
	bool saveAs(const std::string & filename);
	bool revert();
	bool close();
	
	virtual bool shouldSave();
	virtual bool confirmDiscardChanges();
	virtual std::string chooseOpenFileName();
	virtual std::string chooseSaveFileName();
	virtual void doClear();
	virtual bool doCreate(const std::string & fileName);
	virtual bool doRead(const std::string & fileName);
	virtual bool doWrite(const std::string & fileName);
	virtual void doClose();
	virtual void beforeSave();
	virtual void afterOpen();
	virtual bool doCopy(const std::string & filename);
	
	void setFileName(const std::string & filename);
	std::string fileName() const;
	
	void setOpened();
	void setClosed();
	bool isOpened() const;
	bool isUntitled() const;
	
	void setDirty();
	void setClean();
	bool isDirty() const;
	
	void setLatestError(BaseFile::ErrorMsg err);
	BaseFile::ErrorMsg latestError() const;
	
	bool isReverting() const;
	
	static bool FileExists(const std::string & name);
	
private:
	enum ClearMode {
		Normal = 0,
		Revert = 1
	};
	
	std::string m_fileName;
	ErrorMsg m_error;
	ClearMode m_clearMode;
	
	bool _opened;
	bool m_dirty;
};