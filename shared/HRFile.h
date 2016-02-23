/*
 *  HRFile.h
 *  mallard
 *
 *  Created by jian zhang on 10/16/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <HFile.h>
namespace aphid {

class HRFile : public HFile {
public:
	HRFile();
	HRFile(const char * name);
	
	virtual bool doRead(const std::string & fileName);
		
protected:

private:

};

}