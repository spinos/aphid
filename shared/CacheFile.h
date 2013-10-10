/*
 *  CacheFile.h
 *  mallard
 *
 *  Created by jian zhang on 10/10/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
 
#pragma once
#include <BaseFile.h>
#include <BaseState.h>

class CacheFile : public BaseFile, public BaseState {
public:
    CacheFile();
	CacheFile(const char * name);
    
	virtual bool open(const std::string & fileName);
	virtual bool save();
	virtual bool close();
private:

};

