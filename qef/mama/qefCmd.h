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

class QefCmd : public MPxCommand 
{
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
    void writeMesh(const MDagPath & path);
    
private:
	enum WorkMode {
		WHelp = 0,
		WCreate = 1
	};
	WorkMode m_mode;
	
};