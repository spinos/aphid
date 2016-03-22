/*
 *  rotaCmd.cpp
 *  rotaConstraint
 *
 *  Created by jian zhang on 7/7/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "QefCmd.h"
#include <maya/MGlobal.h>

QefCmd::QefCmd() {}
QefCmd::~QefCmd() {}

void* QefCmd::creator()
{
	return new QefCmd;
}

MSyntax QefCmd::newSyntax() 
{
	MSyntax syntax;

	syntax.addFlag("-h", "-help", MSyntax::kNoArg);
	syntax.enableQuery(false);
	syntax.enableEdit(false);

	return syntax;
}

MStatus QefCmd::parseArgs(const MArgList &args)
{
	MStatus			ReturnStatus;
	MArgDatabase	argData(syntax(), args, &ReturnStatus);

	if ( ReturnStatus.error() )
		return MS::kFailure;

	m_mode = WCreate;
	
	if(argData.isFlagSet("-h")) m_mode = WHelp;
	
	return MS::kSuccess;
}

MStatus QefCmd::doIt(const MArgList &argList)
{
	MStatus status;
	status = parseArgs(argList);
	if (!status)
		return status;
	
	return printHelp();
}

MStatus QefCmd::printHelp()
{
	MGlobal::displayInfo(MString("Qef help info:\n write polygonal mesh(es) into a staging file.")
		+MString("\n howto use qef cmd:")
		+MString("\n select a group of meshes")
		+MString("\n run command qef"));
	
	return MS::kSuccess;
}
