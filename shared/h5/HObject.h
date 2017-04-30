#ifndef APH_H_OBJECT_H
#define APH_H_OBJECT_H

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

namespace aphid {

class HObject {
public:
	HObject(const std::string & path);
	virtual ~HObject() {}
	
	virtual char validate();
	virtual char create();
	virtual char open();
	virtual void close();
	virtual int objectType() const;
	virtual char exists();
	
	std::string lastName() const;
	std::string parentName() const;

	static std::string ValidPathName(const std::string & name);
	static std::string FullPath(const std::string & entryName, const std::string & sliceName);
	static std::string PartialPath(const std::string & entryName, const std::string & sliceName);
	
	std::string pathToObject() const;
	
	static HDocument FileIO;
	
	hid_t fObjectId;
	std::string fObjectPath;
};

}
#endif        //  #ifndef HOBJECT_H

