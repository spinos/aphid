#ifndef BASELOG_H
#define BASELOG_H

/*
 *  BaseLog.h
 *  testnarrowpahse
 *
 *  Created by jian zhang on 3/29/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
class BaseLog {
public:
	BaseLog(const std::string & fileName);
	virtual ~BaseLog();
	
	void write(const std::string & os);
	void writeTime();
protected:

private:
	std::ofstream m_file;
};
#endif        //  #ifndef BASELOG_H

