/*
 *  LoftMesh.cpp
 *
 *  by connecting a number of profiles   
 *
 *  Created by jian zhang on 8/20/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "LoftMesh.h"

namespace aphid {

LoftMeshBuilder::LoftMeshBuilder()
{
	m_profileBegins.push_back(0);
	m_defaultNml.set(0.f,0.f,1.f);
	m_defaultCol.set(.1f,.7f,.4f);
}

LoftMeshBuilder::~LoftMeshBuilder()
{
	m_points.clear();
	m_profileVertices.clear();
	m_profileBegins.clear();
	m_triangles.clear();
}

void LoftMeshBuilder::addPoint(const Vector3F& v)
{ m_points.push_back(v); }

void LoftMeshBuilder::addProfile(int nv, const int* vs)
{
	m_profileBegins.push_back(nv + m_profileBegins.back() );
	for(int i=0;i<nv;++i) {
		m_profileVertices.push_back(vs[i]);
	}
}

void LoftMeshBuilder::connectProfiles(int a, int b, bool isEven)
{
	const int anv = m_profileBegins[a + 1] - m_profileBegins[a];
	const int bnv = m_profileBegins[b + 1] - m_profileBegins[b];
	
	int ia = m_profileBegins[a];
	int ib = m_profileBegins[b];
	
	for(int i=0;i<anv-1;++i) {
	
		if(i > bnv-1) return;
		
		if(isEven) 
			addEvenTriangles(m_profileVertices[ia], m_profileVertices[ia + 1], 
				m_profileVertices[ib], m_profileVertices[ib + 1]);
		else 
			addOddTriangles(m_profileVertices[ia], m_profileVertices[ia + 1], 
				m_profileVertices[ib], m_profileVertices[ib + 1]);
		
		ia++;
		ib++;
	}
	
}

void LoftMeshBuilder::addEvenTriangles(int a, int b, int c, int d)
{
	if(a != c) {
		m_triangles.push_back(a);
		m_triangles.push_back(d);
		m_triangles.push_back(c);
	}
	
	if(b != d) {
		m_triangles.push_back(a);
		m_triangles.push_back(b);
		m_triangles.push_back(d);
	}
}

void LoftMeshBuilder::addOddTriangles(int a, int b, int c, int d)
{
	if(a != c) {
		m_triangles.push_back(a);
		m_triangles.push_back(b);
		m_triangles.push_back(c);
	}
	
	if(b != d) {
		m_triangles.push_back(b);
		m_triangles.push_back(d);
		m_triangles.push_back(c);
	}
}

int LoftMeshBuilder::numTriangles() const
{ return m_triangles.size() / 3; }

int LoftMeshBuilder::numPoints() const
{ return m_points.size(); }

int LoftMeshBuilder::numProfiles() const
{ return m_profileBegins.size() - 1; }

int LoftMeshBuilder::numProfileVertices() const
{ return m_profileVertices.size(); }

int LoftMeshBuilder::numProfileBegins() const
{ return m_profileBegins.size(); }

int LoftMeshBuilder::getProfileVertex(int i) const
{ return m_profileVertices[i]; }

int LoftMeshBuilder::getProfileBegin(int i) const
{ return m_profileBegins[i]; }

void LoftMeshBuilder::getPoint(Vector3F& dst, int i) const
{ dst = m_points[i]; }

void LoftMeshBuilder::getTriangle(int* dst, int i) const
{ 
	dst[0] = m_triangles[i * 3]; 
	dst[1] = m_triangles[i * 3 + 1]; 
	dst[2] = m_triangles[i * 3 + 2]; 
}

const Vector3F& LoftMeshBuilder::defaultNormal() const
{ return m_defaultNml; }

const Vector3F& LoftMeshBuilder::defaultColor() const
{ return m_defaultCol; }

void LoftMeshBuilder::projectTexcoord(ATriangleMesh* msh,
						BoundingBox& bbx) const
{
	bbx = msh->calculateGeomBBox();
	const float width = bbx.distance(0);
	const float height = bbx.distance(1);
	const float woverh = width / height;
	float sr;
	if(woverh > 1.f) {
		sr = .995f / width;
	} else {
		sr = .995f / height;
	}
	const Vector3F offset = bbx.getMin();
	const float xoffset = -offset.x;
	const float yoffset = -offset.y;
	
	float * texc = msh->triangleTexcoords();
	const int nt = msh->numTriangles();
	Vector3F * p = msh->points();
	unsigned * ind = msh->indices();
	
	int acc=0;
	for(int i=0;i<nt;++i) {
		const int i3 = i * 3;
		for(int j=0;j<3;++j) {
			
			const Vector3F& pj = p[ind[i3 + j] ];
			texc[acc++] = .0025f + (pj.x + xoffset) * sr;
			texc[acc++] = .0025f + (pj.y + yoffset) * sr;
		}
	}
}

LoftMesh::LoftMesh()
{}

LoftMesh::~LoftMesh()
{}

void LoftMesh::createMesh(const LoftMeshBuilder& builder )
{
	const int nv = builder.numPoints();
	const int ntri = builder.numTriangles();
	create(nv, ntri);
	createVertexColors(nv);
	
	Vector3F * pv = points();
	for(int i=0;i<nv;++i) {
		builder.getPoint(pv[i], i);
	}
	
	int* triind = (int*)indices();
	for(int i=0;i<ntri;++i) {
		builder.getTriangle(&triind[i * 3], i);
	}
	
	Vector3F * nmlDst = vertexNormals();
	Vector3F * colDst = (Vector3F *)vertexColors();
	for(int i=0;i<nv;++i) {
		nmlDst[i] = builder.defaultNormal();
		colDst[i] = builder.defaultColor();
	}
	
	BoundingBox bbx;
	builder.projectTexcoord(this, bbx);
	m_width = bbx.distance(0);
	m_height = bbx.distance(1);
	m_depth = bbx.distance(2);
	
	const int npv = builder.numProfileVertices();
	const int npb = builder.numProfileBegins();
	
	m_profileVertices.reset(new int[npv]);
	for(int i=0;i<npv;++i)
		m_profileVertices[i] = builder.getProfileVertex(i);
		
	m_profileBegins.reset(new int[npb]);
	for(int i=0;i<npb;++i)
		m_profileBegins[i] = builder.getProfileBegin(i);
		
}

const float& LoftMesh::width() const
{ return m_width; }

const float& LoftMesh::height() const
{ return m_height; }

const float& LoftMesh::depth() const
{ return m_depth; }

float LoftMesh::widthHeightRatio() const
{ return width() / height(); }

void LoftMesh::getProfileRange(int& vbegin, int& vend,
			const int& i) const
{
	vbegin = m_profileBegins[i];
	vend = m_profileBegins[i+1];
}

const int& LoftMesh::getProfileVertex(const int& i) const
{ return m_profileVertices[i]; }

}