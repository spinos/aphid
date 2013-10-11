#ifndef HOBJECT_H
#define HOBJECT_H

/*
 *  HObject.h
 *  helloHdf
 *
 *  Created by jian zhang on 6/12/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include "HDocument.h"
#include <string>

class HObject {
public:
	HObject(const std::string & path);
	virtual ~HObject() {}
	
	virtual char validate();
	virtual char create();
	virtual char open();
	virtual void close() {}
	virtual int objectType() const;
	virtual char exists();

	static std::string ValidPathName(const std::string & name);
	static std::string FullPath(const std::string & entryName, const std::string & sliceName);
	
	std::string pathToObject() const;
	
	static HDocument FileIO;
	
	hid_t fObjectId;
	std::string fObjectPath;
};
#endif        //  #ifndef HOBJECT_H

