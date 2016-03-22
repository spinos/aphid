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
#include <ASearchHelper.h>

using namespace aphid;

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
	
	if(m_mode == WCreate) return writeSelected();
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

MStatus QefCmd::writeSelected()
{
    MSelectionList sels;
 	MGlobal::getActiveSelectionList( sels );
	
	if(sels.length() < 1) {
		MGlobal::displayWarning("proxyPaintTool wrong selection, select mesh(es) and a viz to connect");
		return MS::kFailure;
	}
    
    MStatus stat;
    MItSelectionList transIter(sels, MFn::kTransform, &stat);
	if(!stat) {
		MGlobal::displayWarning("qef no group selected");
		return MS::kFailure;
	}
	
    std::map<std::string, MDagPath> orderedMeshes;
    for(;!transIter.isDone(); transIter.next() ) {
		MDagPath pt;
		transIter.getDagPath(pt);
		
        MDagPathArray meshes;
        ASearchHelper::LsAllTypedPaths(meshes, pt, MFn::kMesh);
        ASearchHelper::LsAll(orderedMeshes, meshes);
	}
    
    if(orderedMeshes.size() < 1) {
        MGlobal::displayWarning("qef no mesh selected");
		return MS::kFailure;
    }
    
    std::map<std::string, MDagPath>::const_iterator meshIt = orderedMeshes.begin();
    for(;meshIt != orderedMeshes.end(); ++meshIt) {
        writeMesh(meshIt->second);
    }
    
    return MS::kSuccess;
}

void QefCmd::writeMesh(const MDagPath & path)
{
    AHelper::Info<MString>("w mesh", path.fullPathName() );
    MStatus stat;
    MIntArray vertices;
    int ntris, i, totalNTri = 0, nv;
    int triV[3];
    MItMeshPolygon faceIt(path);
    for(i=0; !faceIt.isDone(); faceIt.next(), i++) {
        faceIt.numTriangles(ntris);
        totalNTri += ntris;
        
        faceIt.getVertices(vertices);
        nv = vertices.lenght();
        
        for(i=1; i<nv-1; ++i ) {
            triV[0] = vertices[0];
            triV[1] = vertices[i];
            triV[2] = vertices[i+1];
        }
        
        AHelper::Info<int>(" tri ", triV[0]);
        AHelper::Info<int>(" tri ", triV[1]);
        AHelper::Info<int>(" tri ", triV[2]);
    }
    
    AHelper::Info<int>(" total n tri ", totalNTri);
}
