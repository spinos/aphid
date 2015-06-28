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
#include <maya/MMatrix.h>
class HesperisFile;
class HesperisIO {
public:
	static bool WriteCurves(MDagPathArray & paths, HesperisFile * file);
	static bool IsCurveValid(const MDagPath & path);
	static bool WriteMeshes(MDagPathArray & paths, HesperisFile * file);
    static MMatrix GetParentTransform(const MDagPath & path);
    static MMatrix GetWorldTransform(const MDagPath & path);
};