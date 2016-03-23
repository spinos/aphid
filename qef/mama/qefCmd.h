/*
 *  rotaCmd.h
 *  rotaConstraint
 *
 *  Created by jian zhang on 7/7/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <maya/MSyntax.h>
#include <maya/MPxCommand.h> 
#include <maya/MArgDatabase.h>
#include <maya/MDagPath.h>
#include <NTreeIO.h>

using namespace aphid;

class QefCmd : public MPxCommand 
{
	NTreeIO m_io;
	
public:
	QefCmd();
	virtual ~QefCmd();

	virtual MStatus		doIt(const MArgList &argList);
	static MSyntax newSyntax();
	static  void* creator();
	
protected:
	virtual MStatus			parseArgs(const MArgList &argList);
	MStatus printHelp();
	MStatus writeSelected();
	void updateMeshBBox(BoundingBox & bbox, const MDagPath & path);
    int writeMesh(HTriangleAsset & asset, 
					const Vector3F & ref, const MDagPath & path);
    
private:
	enum WorkMode {
		WHelp = 0,
		WCreate = 1
	};
	WorkMode m_mode;
	std::string m_filename;
};