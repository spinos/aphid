/*
 *  MeshSampler.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 12/28/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "MeshSampler.h"
#include <maya/MFloatPointArray.h>
#include <maya/MFloatArray.h>
#include <maya/MFloatMatrix.h>
#include <maya/MItMeshPolygon.h>
#include <mama/AHelper.h>
#include <mama/ASearchHelper.h>
#include <geom/ConvexShape.h>
#include <geom/ATriangleMesh.h>
#include <sdb/VectorArray.h>
#include <geom/ConvexShape.h>
#include <maya/MRenderUtil.h>

namespace aphid {

MeshSampler::MeshSampler()
{}

void MeshSampler::SampleMeshTrianglesInGroup(sdb::VectorArray<cvx::Triangle> & tris,
								BoundingBox & bbox,
							const MDagPath & groupPath)
{
    MDagPathArray meshPaths;
	ASearchHelper::LsAllTypedPaths(meshPaths, groupPath, MFn::kMesh);
	const int n = meshPaths.length();
	if(n < 1) {
		AHelper::Info<MString>("find no mesh in group", groupPath.fullPathName() );
		return;
	}
	
	ASearchHelper scher;
	
	for(int i=0;i<n;++i) {
		const int triBegin = tris.size();
		const MDagPath & curMeshP = meshPaths[i];
	    GetMeshTriangles(tris, bbox, curMeshP, groupPath );
		const int triEnd = tris.size();
		MObject curMeshO = curMeshP.node();
		
		MObject sgo = scher.findShadingEngine(curMeshO);
		if(sgo == MObject::kNullObj) {
			AHelper::Info<MString>(" ERROR mesh has no shading engine", curMeshP.fullPathName() );
			continue;
		}
		SampleTriangles(tris, triBegin, triEnd, sgo);
		
	}
}

bool MeshSampler::SampleTriangles(sdb::VectorArray<cvx::Triangle> & tris,
						const int & iBegin, const int & iEnd,
						const MObject & shadingEngineNode)
{
	const MString sgName = MFnDependencyNode(shadingEngineNode).name();
	const int numSamples = iEnd - iBegin;
	
	MFloatPointArray points;
	MFloatArray uCoord, vCoord;
	float uvContribs[3] = {.333f, .333f, .333f};
	
	for(int i=iBegin;i<iEnd;++i) {
		const cvx::Triangle * atri = tris[i];
		const Vector3F tric = atri->center();
		
		MFloatPoint apnt(tric.x, tric.y, tric.z);
		points.append(apnt);
		
		Float2 auv = atri->interpolateTexcoord(uvContribs);
		uCoord.append(auv.x);
		vCoord.append(auv.y);
		
	}
	
	MFloatMatrix cameraMat;
	cameraMat.setToIdentity();
	
	MFloatVectorArray colors, transps;
	
	MStatus stat = MRenderUtil::sampleShadingNetwork( 
			sgName, 
			numSamples,
			false,
			false,
			cameraMat,
			&points,
			&uCoord,
			&vCoord,
			NULL,
			&points,
			NULL,
			NULL,
			NULL,	// don't need filterSize

			colors,
			transps );
			
	if(stat != MS::kSuccess) {
		AHelper::Info<MString>(" ERROR MeshSampler::SampleTriangles cannot sample shading network", sgName);
		return false;
	}
	
	for(int i=iBegin;i<iEnd;++i) {
		cvx::Triangle * atri = tris[i];
		atri->resetNC();
		
		const MFloatVector & fvc = colors[i-iBegin];
		Vector3F vc(fvc.x, fvc.y, fvc.z);
		for(int j=0;j<3;++j) {
			atri->setC(vc, j);
		}
	}
	
	AHelper::Info<MString>("MeshSampler::SampleTriangles done sample shading network", sgName);
		
	return true;
}

}