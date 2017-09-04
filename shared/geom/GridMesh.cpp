/*
 *  GridMesh.cpp
 *
 *  mesh forms a uniform grid 
 *  originate at zero along xy plane facing +z
 *
 *  Created by jian zhang on 8/9/17.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
 
#include "GridMesh.h"
#include "PlanarTexcoordProjector.h"

namespace aphid {

GridMesh::GridMesh()
{}

GridMesh::GridMesh(int nu, int nv, float du, float dv)
{
	createGrid(nu, nv, du, dv);
}

GridMesh::~GridMesh()
{}

void GridMesh::createGrid(int nu, int nv, float du, float dv)
{
	const int np = (nu + 1) * (nv + 1);
	const int nt = nu * nv * 2;
	create(np, nt);
	
	Vector3F * p = points();
	unsigned * ind = indices();

	int acc = 0;
	for(int j=0;j<=nv;++j) {
		for(int i=0;i<=nu;++i) {
			p[acc++].set(du * i, dv * j, 0.f);
		}
	}

	const int nu1 = nu + 1;
	int itri = 0;
	for(int j=0;j<nv;++j) {
		bool isOdd = j & 1;
		
		for(int i=0;i<nu;++i) {
		
			if(isOdd) 
				addOddCell(ind, itri, i, j, nu1);
			else
				addEvenCell(ind, itri, i, j, nu1);
			
			isOdd = !isOdd; 
		}
	}
	
	m_nu = nu;
	m_nv = nv;
	m_du = du;
	m_dv = dv;
	
	PlanarTexcoordProjector proj;
	proj.setTexcoordOrigin(PlanarTexcoordProjector::tCenteredBox);
	BoundingBox bbx;
	proj.projectTexcoord(this, bbx);
	
	Vector3F * nmlDst = vertexNormals();
	createVertexColors(np);
	Vector3F * colDst = (Vector3F *)vertexColors();
	for(int i=0;i<np;++i) {
		nmlDst[i].set(0.f,0.f,1.f);
		colDst[i].set(.1f,.7f,.4f);
	}
	
}

const int& GridMesh::nu() const
{ return m_nu; }

const int& GridMesh::nv() const
{ return m_nv; }

/// j2 --- i1j2
///  |  \  |     even
/// j1 --- i1j1  
///  |  /  |     odd
/// i0 --- i1  

void GridMesh::addOddCell(unsigned* ind, int& tri,
				const int& i, const int& j,
				const int& nu1)
{
	const int i1 = i + 1;
	const int j1 = j + 1;
	int tri3 = tri * 3;
	ind[tri3    ] = j * nu1 + i;
	ind[tri3 + 1] = j * nu1 + i1;
	ind[tri3 + 2] = j1 * nu1 + i1;
	tri++;
	tri3 = tri * 3;
	ind[tri3    ] = j * nu1 + i;
	ind[tri3 + 1] = j1 * nu1 + i1;
	ind[tri3 + 2] = j1 * nu1 + i;
	tri++;
}

void GridMesh::addEvenCell(unsigned* ind, int& tri,
				const int& i, const int& j,
				const int& nu1)
{
	const int i1 = i + 1;
	const int j1 = j + 1;
	int tri3 = tri * 3;
	ind[tri3    ] = j * nu1 + i;
	ind[tri3 + 1] = j * nu1 + i1;
	ind[tri3 + 2] = j1 * nu1 + i;
	tri++;
	tri3 = tri * 3;
	ind[tri3    ] = j * nu1 + i1;
	ind[tri3 + 1] = j1 * nu1 + i1;
	ind[tri3 + 2] = j1 * nu1 + i;
	tri++;
}

float GridMesh::widthHeightRatio() const
{ return width() / height(); }

float GridMesh::width() const
{ return (m_du * m_nu); }

float GridMesh::height() const
{ return (m_dv * m_nv); }
/*
void GridMesh::projectTexcoord()
{
	float sv;
	if(widthHeightRatio() > 1.f) {
		sv = .995f / width();
	} else {
		sv = .995f / height();
	}
	
	float * texc = triangleTexcoords();
	const int nt = numTriangles();
	Vector3F * p = points();
	unsigned * ind = indices();
	
	int acc=0;
	for(int i=0;i<nt;++i) {
		const int i3 = i * 3;
		for(int j=0;j<3;++j) {
			
			const Vector3F& pj = p[ind[i3 + j] ];
			texc[acc++] = .0025f + pj.x * sv;
			texc[acc++] = .0025f + pj.y * sv;
		}
	}
	
}
*/
int GridMesh::GetNumVerticesPerRow(const ATriangleMesh* msh)
{
	const unsigned * ind = msh->indices();
	return ind[2];
}

}