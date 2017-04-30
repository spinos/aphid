/*
 *  UntangleCmd.h
 *  manuka
 *
 *  Created by jian zhang on 1/22/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <maya/MPxCommand.h>
#include <maya/MSyntax.h>
#include <maya/MDagPath.h>
#include <maya/MPoint.h>
#include <maya/MSelectionList.h>
#include <vector>
#include <map>

class UntangleCmd : public MPxCommand
{
public:
					UntangleCmd(); 
	virtual			~UntangleCmd(); 
	static void*	creator();

	MStatus			doIt(const MArgList& args);
	MStatus			parseArgs(const MArgList& args);
	static MSyntax	newSyntax();

private:
	void addTriangles(std::vector<int> & connections,
					std::map<int, int > & vertices,
					int vertexIndOffset,
					const MDagPath & mesh, MObject & faces);
	void packVertices(std::map<int, int > & vertices);
	
	void convertToVert(MSelectionList & selVerts, 
					const MDagPath & item, MObject & components);
	MObject createMergeDeformer();
	void initDeformer(const std::vector<int> & connections,
					const std::map<int, int > & vertices, 
					const MObject & deformer);
	
};