/*
 *  BaseFileGroup.h
 *  
 *
 *  Created by jian zhang on 2/26/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <BaseFile.h>
#include <map>

class BaseFileGroup {
public:
	BaseFileGroup();
	virtual ~BaseFileGroup();
	
	void addFile(BaseFile * file);
	bool getFile(const std::string & name, BaseFile * dst);
protected:

private:
	std::map<std::string, BaseFile *> m_files;
};