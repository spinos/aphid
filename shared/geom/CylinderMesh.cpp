/*
 *  CylinderMesh.cpp
 *
 *  mesh forms a uniform grid 
 *  originate at zero along xy plane facing +z
 *
 *  Created by jian zhang on 8/9/17.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
 
#include "CylinderMesh.h"
#include <math/miscfuncs.h>

namespace aphid {

CylinderMesh::CylinderMesh()
{}

CylinderMesh::CylinderMesh(int nu, int nv, float radius, float height)
{
	createCylinder(nu, nv, radius, height);
}

CylinderMesh::~CylinderMesh()
{}

void CylinderMesh::createCylinder(int nu, int nv, float radius, float height)
{
/// uniform segment heights
	const float dv = 1.f / (float)nv;
	float* hs = new float[nv+1];
	for(int i=0;i<=nv;++i) {
		hs[i] = dv * i;
	}
	createCylinder1(nu, nv, radius, height, hs);
	delete[] hs;
}

void CylinderMesh::createCylinder1(int nu, int nv, float radius, float height,
		const float* heightSegs)
{
	const int np = nu * (nv + 1);
	const int nt = nu * nv * 2;
	create(np, nt);
	
	Vector3F * p = points();
	unsigned * ind = indices();
	
	const float da = TWOPIF / (float)nu;
	m_circum = Vector3F(cos(da), 0.f, sin(da) ).distanceTo(Vector3F::XAxis) * radius * nu; 
	const float du = m_circum / (float)nu;

	int acc = 0;
	for(int j=0;j<=nv;++j) {
		for(int i=0;i<nu;++i) {
			p[acc++].set(radius * cos(da * i), heightSegs[j], -radius * sin(da * i));
		}
	}
	
	Vector2F* texc = (Vector2F*)triangleTexcoords();

	int itri = 0;
	for(int j=0;j<nv;++j) {
		bool isOdd = j & 1;
		
		int j1 = j + 1;
		
		for(int i=0;i<nu;++i) {
		
			int i1 = i + 1;
			if(i1 >= nu)
				i1 = 0;
		
			if(isOdd) 
				addOddCell(ind, itri, i, j, i1, j1, nu, du, heightSegs, texc);
			else
				addEvenCell(ind, itri, i, j, i1, j1, nu, du, heightSegs, texc);
			
			isOdd = !isOdd; 
		}
	}
	
	m_nu = nu;
	m_nv = nv;
	m_radius = radius;
	m_height = height;
	projectTexcoord();
	
	Vector3F * nmlDst = vertexNormals();
	createVertexColors(np);
	Vector3F * colDst = (Vector3F *)vertexColors();
	for(int i=0;i<np;++i) {
		nmlDst[i].set(p[i].x ,0.f, p[i].z);
		nmlDst[i].normalize();
		colDst[i].set(.1f,.7f,.4f);
	}
	
}

const int& CylinderMesh::nu() const
{ return m_nu; }

const int& CylinderMesh::nv() const
{ return m_nv; }

/// j2 --- i1j2
///  |  \  |     even
/// j1 --- i1j1  
///  |  /  |     odd
/// i0 --- i1  

void CylinderMesh::addOddCell(unsigned* ind, int& tri,
				const int& i, const int& j,
				const int& i1, const int& j1,
				const int& nu,
				const float& du, const float* heightSegs,
				Vector2F* fvps)
{
	int tri3 = tri * 3;
	ind[tri3    ] = j * nu + i;
	ind[tri3 + 1] = j * nu + i1;
	ind[tri3 + 2] = j1 * nu + i1;
	
	fvps[tri3    ].set(du * i,  heightSegs[j]);
	fvps[tri3 + 1].set(du * (i+1),  heightSegs[j]);
	fvps[tri3 + 2].set(du * (i+1),  heightSegs[j+1]);
	
	tri++;
	tri3 = tri * 3;
	ind[tri3    ] = j * nu + i;
	ind[tri3 + 1] = j1 * nu + i1;
	ind[tri3 + 2] = j1 * nu + i;
	
	fvps[tri3    ].set(du * i,  heightSegs[j]);
	fvps[tri3 + 1].set(du * (i+1),  heightSegs[j+1]);
	fvps[tri3 + 2].set(du * i,  heightSegs[j+1]);
	
	tri++;
}

void CylinderMesh::addEvenCell(unsigned* ind, int& tri,
				const int& i, const int& j,
				const int& i1, const int& j1,
				const int& nu,
				const float& du, const float* heightSegs,
				Vector2F* fvps)
{
	int tri3 = tri * 3;
	ind[tri3    ] = j * nu + i;
	ind[tri3 + 1] = j * nu + i1;
	ind[tri3 + 2] = j1 * nu + i;
	
	fvps[tri3    ].set(du * i,  heightSegs[j]);
	fvps[tri3 + 1].set(du * (i+1),  heightSegs[j]);
	fvps[tri3 + 2].set(du * i,  heightSegs[j+1]);
	
	tri++;
	tri3 = tri * 3;
	ind[tri3    ] = j * nu + i1;
	ind[tri3 + 1] = j1 * nu + i1;
	ind[tri3 + 2] = j1 * nu + i;
	
	fvps[tri3    ].set(du * (i+1),  heightSegs[j]);
	fvps[tri3 + 1].set(du * (i+1),  heightSegs[j+1]);
	fvps[tri3 + 2].set(du * i,  heightSegs[j+1]);
	
	tri++;
}

const float& CylinderMesh::radius() const
{ return m_radius; }

const float& CylinderMesh::circumference() const
{ return m_circum; }

const float& CylinderMesh::height() const
{ return m_height; }

float CylinderMesh::circumferenceHeightRatio() const
{ return circumference() / height(); }

void CylinderMesh::projectTexcoord()
{
	float sv;
	if(circumferenceHeightRatio() > 1.f) {
		sv = .995f / circumference();
	} else {
		sv = .995f / height();
	}
	
	const Vector2F ori(.0025f, .0025f);
	
	Vector2F* texc = (Vector2F*)triangleTexcoords();
	const int n = numTriangles() * 3;

	for(int i=0;i<n;++i) {
		texc[i] *= sv;
		texc[i] += ori;
		
	}
	
}

}