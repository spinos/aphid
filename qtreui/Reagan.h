/*
 *  Reagan.h
 *  reui
 *
 *  Created by jian zhang on 12/15/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef _Reagan_H
#define _Reagan_H
#include <string>

class Reagan {
public:
	Reagan();
	~Reagan();
	static void runReReplace(std::string &result, const std::string &pattern, const std::string &format);
	static void validateUnixPath(std::string &result);
	static void removeNamespaceInFullPathName(std::string &result);
	
};
#endif