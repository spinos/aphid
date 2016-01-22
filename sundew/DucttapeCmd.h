/*
 *  DucttapeCmd.h
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

class DucttapeCmd : public MPxCommand
{
public:
					DucttapeCmd(); 
	virtual			~DucttapeCmd(); 
	static void*	creator();

	MStatus			doIt(const MArgList& args);
	MStatus			parseArgs(const MArgList& args);
	static MSyntax	newSyntax();

private:
	struct IndexPoint {
		MPoint _pnt;
		int _ind;
	};
	void addFaces(std::vector<int> & counts, 
					std::vector<int> & connections,
					std::map<int, IndexPoint > & vertices,
					int vertexIndOffset,
					const MDagPath & mesh, MObject & faces);
	void packVertices(std::map<int, IndexPoint > & vertices);
	MDagPath buildMesh(const std::vector<int> & counts, 
					const std::vector<int> & connections,
					std::map<int, IndexPoint > & vertices,
					MStatus * stat);
	void convertToVert(MSelectionList & selVerts, 
					const MDagPath & item, MObject & components);
	MObject createMergeDeformer();
	void connectBranch(const MObject & branch, const MObject & merge);
};