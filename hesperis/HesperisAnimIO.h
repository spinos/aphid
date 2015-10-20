/*
 *  HesperisAnimIO.h
 *  opium
 *
 *  Created by jian zhang on 10/21/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "HesperisIO.h"

class HesperisAnimIO : public HesperisIO {
public:
	static bool WriteAnimation(const MPlug & attrib, const MObject & animCurveObj, double secondsPerFrame,
								const std::string & beheadName = "");
};