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
#include <mama/AHelper.h>
#include <mama/ASearchHelper.h>
#include <geom/ConvexShape.h>
#include <geom/ATriangleMesh.h>
#include <sdb/VectorArray.h>
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

void MeshHelper::UpdateMeshTriangleUVs(ATriangleMesh * trimesh,
						const MObject & meshNode)
{
	MStatus stat;
	MItMeshPolygon faceIter(meshNode, &stat);
	if(!stat) {
		return;
	}
	
	if(!faceIter.hasUVs() ) {
	    AHelper::Info<int>(" WARNING mesh has no uv", 0 );
	     return;   
	}
	
	Float2 tuvs[3];
	int ti = 0;
	MFloatArray uArray, vArray;
	for(;!faceIter.isDone();faceIter.next() ) {
		faceIter.getUVs(uArray, vArray);
		tuvs[0].set(uArray[0], vArray[0]);
		const int nt = uArray.length() - 2;
		for(int i=0;i<nt;++i) {
			tuvs[1].set(uArray[i+1], vArray[i+1]);
			tuvs[2].set(uArray[i+2], vArray[i+2]);
			trimesh->setTriangleTexcoord(ti, tuvs);
			ti++;
		}
	}
	
}

void MeshHelper::GetMeshTriangles(sdb::VectorArray<cvx::Triangle> & tris,
								BoundingBox & bbox,
								const MDagPath & meshPath,
								const MDagPath & tansformPath)
{
	AHelper::Info<MString>("get mesh triangles", meshPath.fullPathName() );
	
	MMatrix worldTm = AHelper::GetWorldParentTransformMatrix2(meshPath, tansformPath);
	
    MStatus stat;
	
    MIntArray vertices;
    int i, j, nv;
	MPoint dp[3];
	Vector3F fp[3];
	Float2 tuvs[3];
	int tvert[3];
	MFloatArray uArray, vArray;
	cvx::Triangle tri;
	
	MItMeshPolygon faceIt(meshPath);
	
	const bool hasUv = faceIt.hasUVs();
	
    for(; !faceIt.isDone(); faceIt.next() ) {

		faceIt.getVertices(vertices);
        nv = vertices.length();
        
		dp[0] = faceIt.point(0, MSpace::kObject );
		dp[0] *= worldTm;
		
		tvert[0] = vertices[0];
		
		if(hasUv) {
			faceIt.getUVs(uArray, vArray);
			tuvs[0].set(uArray[0], vArray[0]);
		}
		
        for(i=1; i<nv-1; ++i ) {
			dp[1] = faceIt.point(i, MSpace::kObject );
			dp[2] = faceIt.point(i+1, MSpace::kObject );
			
			dp[1] *= worldTm;	
			dp[2] *= worldTm;
			
			tvert[1] = vertices[i];
			tvert[2] = vertices[i+1];
			
			if(hasUv) {
				tuvs[1].set(uArray[i], vArray[i]);
				tuvs[2].set(uArray[i+1], vArray[i+1]);
				tri.setUVs(tuvs);
			}
			
			for(j=0; j<3; ++j) {
				fp[j].set(dp[j].x, dp[j].y, dp[j].z);
				tri.setP(fp[j], j);
				tri.setInd(tvert[j], j);
				bbox.expandBy(fp[j], 1e-4f);
			}
			
			tris.insert(tri);
        }
    }
}

void MeshHelper::GetMeshTrianglesInGroup(sdb::VectorArray<cvx::Triangle> & tris,
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
	
	for(int i=0;i<n;++i) {
	    GetMeshTriangles(tris, bbox, meshPaths[i], groupPath );
	}
}

MObject MeshHelper::CreateMesh(const ATriangleMesh & msh,
					MObject parent, CreateProfile * prof)
{
	const int numVertices = msh.numPoints();
	const int numPolygons = msh.numTriangles();
	const int ninds = numPolygons * 3;
	
	MPointArray vertexArray;
	vertexArray.setLength(numVertices);
	
	const Vector3F * pos = msh.points();
	for(int i=0;i<numVertices;++i) {
		vertexArray[i] = MPoint(pos[i].x, pos[i].y, pos[i].z);
	}
	
	MIntArray polygonCounts;
	polygonCounts.setLength(numPolygons);
	for(int i=0;i<numPolygons;++i) {
		polygonCounts[i] = 3;
	}
	
	MIntArray polygonConnects;
	polygonConnects.setLength(ninds);
	
	const unsigned * ind = msh.indices();
	for(int i=0;i<ninds;++i) {
		polygonConnects[i] = ind[i];
	}
	
	MFnMesh mf;
	MObject node = mf.create (numVertices, numPolygons, 
		vertexArray, 
		polygonCounts, 
		polygonConnects, 
		parent );
		
	if(!prof) {
		return node;
	}
	
	if(prof->_hasUV) {
		MFloatArray uArray, vArray;
		MIntArray uvIds;
		MIntArray uvCounts;
		MString uvSet("map1");
		
		int ci = 0;
		for(int i=0;i<numPolygons;++i) {
			const Float2 * triuv = msh.triangleTexcoord(i);
			uArray.append(triuv[0].x);
			vArray.append(triuv[0].y);
			uArray.append(triuv[1].x);
			vArray.append(triuv[1].y);
			uArray.append(triuv[2].x);
			vArray.append(triuv[2].y);
			
			uvIds.append(ci);
			uvIds.append(ci + 1);
			uvIds.append(ci + 2);
			ci += 3;
			
			uvCounts.append(3);
			
		}
		
		MStatus ms = mf.setUVs( uArray, vArray, &uvSet);
		if(!ms)
			MGlobal::displayWarning(MString(" MeshHelper cannot create uv set coord ")+uvSet);
		
		ms = mf.assignUVs( uvCounts, uvIds, &uvSet );
		if(!ms)
			MGlobal::displayWarning(MString(" MeshHelper cannot create uv set uvid ")+uvSet);
		
	}
		
	return node;
}

}