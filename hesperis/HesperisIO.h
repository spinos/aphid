/*
 *  HesperisIO.h
 *  hesperis
 *
 *  Created by jian zhang on 4/21/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <maya/MDagPath.h>
#include <maya/MDagPathArray.h>
class HesperisIO {
public:
	static bool WriteCurves(MDagPathArray & paths, const std::string & fileName);
	static bool IsCurveValid(const MDagPath & path);
};