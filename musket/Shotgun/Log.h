/*
 *  Log.h
 *  shotgunAPI
 *
 *  Created by jian zhang on 3/19/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <Shotgun/types.h>

namespace SG {
class Log {
public:
	Log();
	static void print(const xmlrpc_c::paramList & params);
	static void printValues(const std::map<std::string, xmlrpc_c::value> & values);
	static void printValues(const std::vector<xmlrpc_c::value> & values);
	static void printValue(const xmlrpc_c::value & value);
};
}