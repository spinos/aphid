/*
 *  StickyCmd.h
 *  hair
 *
 *  Created by jian zhang on 6/3/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <maya/MPxCommand.h>
#include <maya/MSyntax.h>
#include <maya/MDagPath.h>
#include <maya/MPoint.h>

class StickyCmd : public MPxCommand
{
public:
					StickyCmd(); 
	virtual			~StickyCmd(); 
	static void*	creator();

	MStatus			doIt(const MArgList& args);
	MStatus			parseArgs(const MArgList& args);
	static MSyntax	newSyntax();

private:
	void addPosition(MPoint * sum, float * mass, const MDagPath & mesh, MObject & vert);
	void getVertClosestToMean(float *minD, MDagPath & closestMesh, unsigned & closestVert, 
								const MDagPath & mesh, MObject & vert, const MPoint & mean);
	MObject connectViz(unsigned vert);
	MObject createDeformer();
	void connectDeformer(const MObject & viz, const MObject & deformer, int meshId);
	
};
