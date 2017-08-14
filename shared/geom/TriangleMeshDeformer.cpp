/*
 *  TriangleMeshDeformer.cpp
 *  
 *  bend effect > twist effect > roll
 *
 *  Created by jian zhang on 8/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "TriangleMeshDeformer.h"
#include "geom/ATriangleMesh.h"
#include <math/Matrix44F.h>
#include <geom/ConvexShape.h>

namespace aphid {

TriangleMeshDeformer::TriangleMeshDeformer()
{ 
    m_np = 0;
}

TriangleMeshDeformer::~TriangleMeshDeformer()
{}

void TriangleMeshDeformer::setOriginalMesh(const ATriangleMesh * mesh)
{
	if(!mesh) 
	    return;
	
	int np = mesh->numPoints();
	if(m_np < np) {
	    m_points.reset(new Vector3F[np]);
	    m_normals.reset(new Vector3F[np]);
	}
	
	memcpy( m_points.get(), mesh->points(), np * 12 );
	m_np = np;
}

const Vector3F * TriangleMeshDeformer::deformedPoints() const
{ return m_points.get(); }

const Vector3F * TriangleMeshDeformer::deformedNormals() const
{ return m_normals.get(); }

Vector3F* TriangleMeshDeformer::points()
{ return m_points.get(); }

Vector3F* TriangleMeshDeformer::normals()
{ return m_normals.get(); }

const int& TriangleMeshDeformer::numPoints() const
{ return m_np; }

void TriangleMeshDeformer::deform(const ATriangleMesh * mesh)
{}

float TriangleMeshDeformer::getRowMean(int rowBegin, int nv, int& nvRow, float& rowBase ) const
{
	rowBase = m_points[rowBegin].y;
	
	float res = 0.f;
	nvRow = 0;
	for(int i=rowBegin;i<nv;++i) {
		if(m_points[i].y > rowBase + 1e-3f)
			break;
			
		res += m_points[i].y;
		nvRow++;
	}
	res /= (float)nvRow;
	return res;
}

void TriangleMeshDeformer::calculateNormal(const ATriangleMesh * mesh)
{
	const int nv = mesh->numPoints();
	std::memset(m_normals.get(), 0, nv * 12);
	const int ni = mesh->numIndices();
	const unsigned * ind = mesh->indices();
	const Vector3F * ps = m_points.get();
	
	cvx::Triangle atri;
	Vector3F triNm;

	for(int i=0;i<ni;i+=3) {
		const unsigned & i0 = ind[i];
		const unsigned & i1 = ind[i+1];
		const unsigned & i2 = ind[i+2];
		
		const Vector3F & a = ps[i0];
		const Vector3F & b = ps[i1];
		const Vector3F & c = ps[i2];
		
		atri.set(a, b, c);
		triNm = atri.calculateNormal();
		
		m_normals[i0] += triNm;
		m_normals[i1] += triNm;
		m_normals[i2] += triNm;
	}
	
	for(int i=0;i<nv;++i) {
		m_normals[i].normalize();
	}

}

void TriangleMeshDeformer::updateGeom(ATriangleMesh* outMesh,
				const ATriangleMesh* inMesh)
{
	const int np = inMesh->numPoints();
	const int nt = inMesh->numTriangles();
	    
	outMesh->create(np, nt);
	outMesh->createVertexColors(np);
	
	unsigned * indDst = outMesh->indices();
	memcpy(indDst, inMesh->indices(), nt * 12);
	
	float * colDst = outMesh->vertexColors();
	memcpy(colDst, inMesh->vertexColors(), np * 12);
	
	float * texcoordDst = outMesh->triangleTexcoords();
	memcpy(texcoordDst, inMesh->triangleTexcoords(), nt * 24);
	
	Vector3F * pntDst = outMesh->points();
	Vector3F * nmlDst = outMesh->vertexNormals();

	memcpy(pntDst, deformedPoints(), np * 12);
	memcpy(nmlDst, deformedNormals(), np * 12);
}

int TriangleMeshDeformer::GetRowNv(float& rowBase, int rowBegin, int nv, const Vector3F* ps )
{
	rowBase = ps[rowBegin].y;
	
	int nvRow = 1;
	for(int i=rowBegin + 1;i<nv;++i) {
		if(ps[i].y > rowBase + 1e-3f)
			break;
			
		nvRow++;
	}
	return nvRow;
}

int TriangleMeshDeformer::GetNumRows(const ATriangleMesh * mesh)
{
	const int & nv = mesh->numPoints();
	const Vector3F* ps = mesh->points();
	
	float yBase;
/// first row
	int rownv = GetRowNv(yBase, 0, nv, ps);
	float lastYBase = yBase;
	
	int nrow = 1;
	for(int i=rownv;i<nv;i+=rownv) {
		rownv = GetRowNv(yBase, i, nv, ps);
		nrow++;
	}
	return nrow;
}

}
