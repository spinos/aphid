#ifndef BCCGLOBAL_H
#define BCCGLOBAL_H

/*
 *  BccGlobal.h
 *  testbcc
 *
 *  Created by jian zhang on 4/21/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <string>
#include <sstream>
#include <BaseBuffer.h>
#include <GeometryArray.h>
#include <KdIntersection.h>

#define TEST_FIT 1
#define WORLD_USE_FIT 1
#define WORLD_TEST_SINGLE 0

class BccGlobal {
public:
	static std::string FileName;
};
#endif        //  #ifndef BCCGLOBAL_H
