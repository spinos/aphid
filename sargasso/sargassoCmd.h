/*
 *  rotaCmd.h
 *  rotaConstraint
 *
 *  Created by jian zhang on 7/7/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "sargasso_common.h"
#include <maya/MSyntax.h>
#include <maya/MPxCommand.h> 
#include <maya/MArgDatabase.h>

class SargassoCmd : public MPxCommand 
{
public:
	SargassoCmd();
	virtual ~SargassoCmd();

	virtual MStatus		doIt(const MArgList &argList);
	static MSyntax newSyntax();
	static  void* creator();

protected:
	virtual MStatus			parseArgs(const MArgList &argList);
	MStatus printHelp();
	MStatus createNode(const MObjectArray & transforms,
					const MObject & targetMesh);
private:
	enum WorkMode {
		WHelp = 0,
		WCreate = 1
	};
	WorkMode m_mode;
};