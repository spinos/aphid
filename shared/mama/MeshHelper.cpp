/*
 *  MeshHelper.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 12/28/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "MeshHelper.h"
#include <maya/MFnMesh.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MItMeshVertex.h>
#include <AHelper.h>
#include <ASearchHelper.h>
#include <geom/ConvexShape.h>

namespace aphid {

MeshHelper::MeshHelper()
{}

unsigned MeshHelper::GetMeshNv(const MObject & meshNode)
{
    MStatus stat;
    MFnMesh fmesh(meshNode, &stat);
    if(!stat) {
        std::cout<<"AHelper::GetMeshNv no mesh fn to node";
        return 0;   
    }
    return fmesh.numVertices(); 
}

void MeshHelper::CountMeshNv(int & nv,
					const MDagPath & meshPath)
{
	MStatus stat;
	MItMeshVertex vertIt(meshPath, MObject::kNullObj, &stat);
	if(!stat) {
		AHelper::Info<MString>("not a mesh", meshPath.fullPathName() );
		return;
	}
	nv += (int)vertIt.count();
}

void MeshHelper::GetMeshTriangles(MIntArray & triangleVertices,
							const MDagPath & meshPath)
{
	MStatus stat;
	MFnMesh fmesh(meshPath, &stat);
	if(!stat) {
		AHelper::Info<MString>("not a mesh", meshPath.fullPathName() );
		return;
	}
	MIntArray triangleCounts;
	fmesh.getTriangles (triangleCounts, triangleVertices );

}

void MeshHelper::GetMeshTrianglesInGroup(MIntArray & triangleVertices,
							const MDagPath & groupPath)
{
	triangleVertices.clear();
	MDagPathArray meshPaths;
	ASearchHelper::LsAllTypedPaths(meshPaths, groupPath, MFn::kMesh);

	const int n = meshPaths.length();
	if(n < 1) {
		AHelper::Info<MString>("find no mesh in group", groupPath.fullPathName() );
		return;
	}
	
	int * vbegins = new int[n];
	
	int nv = 0;
	for(int i=0;i<n;++i) {
		vbegins[i] = nv;
		CountMeshNv(nv, meshPaths[i]);
	}
	
	for(int i=0;i<n;++i) {
		MIntArray avs;
		GetMeshTriangles(avs, meshPaths[i]);
		
		const int & offset = vbegins[i];
		const int nj = avs.length();
		for(int j=0;j<nj;++j) {
			triangleVertices.append(avs[j] + offset);
		}
		avs.clear();
	}
	
	delete[] vbegins;
}

void MeshHelper::ScatterTriangleVerticesPosition(MVectorArray & pos,
						const float * pnts, const int & np,
						const MIntArray & triangleVertices, const int & nind)
{
	pos.setLength(nind);
	for(int i=0;i<nind;++i) {
		const float * ci = &pnts[triangleVertices[i] * 3];
		pos[i] = MVector(ci[0], ci[1], ci[2]);
	}
}

void MeshHelper::CalculateTriangleVerticesNormal(MVectorArray & nms,
						const float * pnts, const int & np,
						const MIntArray & triangleVertices,
						const int & nind)
{
	Vector3F * pvNm = new Vector3F[np];
	std::memset(pvNm, 0, np * 12);
	
	cvx::Triangle atri;
	Vector3F triNm;

	for(int i=0;i<nind;i+=3) {
		const int i0 = triangleVertices[i];
		const int i1 = triangleVertices[i+1];
		const int i2 = triangleVertices[i+2];
		
		const float * v0 = &pnts[i0 * 3];
		Vector3F a(v0);
		const float * v1 = &pnts[i1 * 3];
		Vector3F b(v1);
		const float * v2 = &pnts[i2 * 3];
		Vector3F c(v2);
		
		atri.set(a, b, c);
		triNm = atri.calculateNormal();
		
		pvNm[i0] += triNm;
		pvNm[i1] += triNm;
		pvNm[i2] += triNm;
	}
	
	for(int i=0;i<np;++i) {
		pvNm[i].normalize();
	}
	
	nms.setLength(nind);
	for(int i=0;i<nind;++i) {
		const Vector3F & ci = pvNm[triangleVertices[i]];
		nms[i] = MVector(ci.x, ci.y, ci.z);
	}
	
	delete[] pvNm;
}

}