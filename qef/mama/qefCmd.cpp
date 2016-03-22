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
#include <maya/MIntArray.h>
#include <maya/MItMeshVertex.h>

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
	syntax.addFlag("-w", "-write", MSyntax::kString);
	syntax.enableQuery(false);
	syntax.enableEdit(false);

	return syntax;
}

MStatus QefCmd::parseArgs(const MArgList &args)
{
	MStatus			stat;
	MArgDatabase	argData(syntax(), args, &stat);

	if ( stat.error() )
		return MS::kFailure;

	m_mode = WHelp;
	
	if(argData.isFlagSet("-w") ) {
		MString ch;
		stat = argData.getFlagArgument("-w", 0, ch);
		if (!stat) {
			stat.perror("-w flag parsing failed.");
			return stat;
		}
		m_filename = ch.asChar();
		m_mode = WCreate;
	}
	if(argData.isFlagSet("-h") ) m_mode = WHelp;
	
	return MS::kSuccess;
}

MStatus QefCmd::doIt(const MArgList &argList)
{
	MStatus status;
	status = parseArgs(argList);
	if (!status)
		return printHelp();
	
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
	if(!m_io.begin(m_filename, HDocument::oCreate ) ) {
		AHelper::Info<std::string >(" NTree IO cannot crate file", m_filename );
		return MS::kFailure;
	}
	
	HTriangleAsset triass("/triangles");
	
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
	
	BoundingBox bbox;
	for(;meshIt != orderedMeshes.end(); ++meshIt) {
        updateMeshBBox(bbox, meshIt->second);
    }

	bbox.round();
	AHelper::Info<BoundingBox>(" bbox ", bbox);
	
	meshIt = orderedMeshes.begin();
    for(;meshIt != orderedMeshes.end(); ++meshIt) {
        writeMesh(triass, bbox.getMin(), meshIt->second);
    }
    
	triass.save();
	triass.close();
	m_io.end();
	
	AHelper::Info<std::string >(" NTree IO finished writing file", m_filename );
    return MS::kSuccess;
}

void QefCmd::updateMeshBBox(BoundingBox & bbox, const MDagPath & path)
{
	MItMeshVertex vertIt(path);
	for(; !vertIt.isDone(); vertIt.next() ) {
		const MPoint pnt = vertIt.position ( MSpace::kWorld );
		Vector3F v(pnt.x, pnt.y, pnt.z);
		bbox.expandBy(v);
	}
}

void QefCmd::writeMesh(HTriangleAsset & asset, 
						const Vector3F & ref, const MDagPath & path)
{
    AHelper::Info<MString>("w mesh", path.fullPathName() );
    MStatus stat;
	
    MIntArray vertices;
    int i, j, totalNTri = 0, nv;
	MPoint dp[3];
	MVector dn[3];
	Vector3F fp[3], fn[3], col[3];
	col[0].set(1.f, 1.f, 1.f);
	col[1].set(1.f, 1.f, 1.f);
	col[2].set(1.f, 1.f, 1.f);
    MItMeshPolygon faceIt(path);
    for(; !faceIt.isDone(); faceIt.next() ) {

        faceIt.getVertices(vertices);
        nv = vertices.length();
        
        for(i=1; i<nv-1; ++i ) {
			dp[0] = faceIt.point(0, MSpace::kWorld );
			dp[1] = faceIt.point(i, MSpace::kWorld );
			dp[2] = faceIt.point(i+1, MSpace::kWorld );
			
			if(AHelper::AreaOfTriangle(dp[0], dp[1], dp[2]) < 1e-5f ) {
				AHelper::Info<int>(" low triangle area ", faceIt.index() );
				continue;
			}
			
			faceIt.getNormal(0 , dn[0], MSpace::kWorld );
			faceIt.getNormal(i , dn[1], MSpace::kWorld );
			faceIt.getNormal(i+1 , dn[2], MSpace::kWorld );
			
			cvx::Triangle tri;
			for(j=0; j<3; ++j) {
				fp[j].set(dp[j].x - ref.x, dp[j].y - ref.y, dp[j].z - ref.z);
				tri.setP(fp[j], j);
				
				fn[j].set(dn[j].x, dn[j].y, dn[j].z);
				tri.setN(fn[j], j);
				
				tri.setC(col[j], j);
			}
			
			asset.insert(tri);
			
			totalNTri++;
        }
    }
    
    AHelper::Info<int>(" total n tri ", totalNTri);
}
