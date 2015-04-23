/*
 *  TetrahedronSystem.cpp
 *  cudabvh
 *
 *  Created by jian zhang on 2/14/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "TetrahedronSystem.h"
#include <BaseBuffer.h>
#include <tetrahedron_math.h>
#include <HTetrahedronMesh.h>

#define PRINT_VICINITY 0
TetrahedronSystem::TetrahedronSystem() :
m_numTetrahedrons(0), m_numPoints(0), m_numTriangles(0)
{
	m_hostX = new BaseBuffer;
	m_hostXi = new BaseBuffer;
	m_hostV = new BaseBuffer;
	m_hostTetrahedronIndices = new BaseBuffer;
	m_hostTriangleIndices = new BaseBuffer;
	m_hostMass = new BaseBuffer;
	m_hostAnchor = new BaseBuffer;
	m_hostTetrahedronVicinityInd = new BaseBuffer;
	m_hostTetrahedronVicinityStart = new BaseBuffer;
	m_totalMass = 10.f;
}

TetrahedronSystem::~TetrahedronSystem() 
{
	delete m_hostX;
	delete m_hostXi;
	delete m_hostV;
	delete m_hostTetrahedronIndices;
	delete m_hostTriangleIndices;
	delete m_hostAnchor;
	delete m_hostTetrahedronVicinityInd;
	delete m_hostTetrahedronVicinityStart;
}

void TetrahedronSystem::generateFromData(TetrahedronMeshData * md)
{
	create(md->m_numTetrahedrons + 100, md->m_numPoints + 400);
	Vector3F * p = (Vector3F *)md->m_pointBuf->data();
	unsigned i;
	for(i=0; i< md->m_numPoints; i++)
	    addPoint((float *)&p[i]);
	
	unsigned * ind = (unsigned *)md->m_indexBuf->data();
	for(i=0; i< md->m_numTetrahedrons; i++)
		addTetrahedron(ind[i*4], ind[i*4+1], ind[i*4+2], ind[i*4+3]);
		
	unsigned * anchor = (unsigned *)md->m_anchorBuf->data();
	for(i=0; i< md->m_numPoints; i++) {
// 0 or 1 for now
		if(anchor[i] > 0) setAnchoredPoint(i, anchor[i]);
	}
}

void TetrahedronSystem::create(const unsigned & maxNumTetrahedrons, const unsigned & maxNumPoints)
{
	m_maxNumTetrahedrons = maxNumTetrahedrons;
	m_maxNumPoints = maxNumPoints;
	m_maxNumTriangles = maxNumTetrahedrons * 4;
	m_hostX->create(m_maxNumPoints * 12);
	m_hostXi->create(m_maxNumPoints * 12);
	m_hostV->create(m_maxNumPoints * 12);
	m_hostMass->create(m_maxNumPoints * 4);
	m_hostAnchor->create(m_maxNumPoints * 4);
	m_hostTetrahedronIndices->create(m_maxNumTetrahedrons * 16);
	m_hostTriangleIndices->create(m_maxNumTriangles * 12);
}

void TetrahedronSystem::setTotalMass(float x)
{ m_totalMass = x; }

void TetrahedronSystem::addPoint(float * src)
{
	if(m_numPoints == m_maxNumPoints) return;
	float * p = &hostX()[m_numPoints * 3];
	float * p0 = &hostXi()[m_numPoints * 3];
	unsigned * anchor = &hostAnchor()[m_numPoints];
	p[0] = src[0];
	p[1] = src[1];
	p[2] = src[2];
	p0[0] = p[0];
	p0[1] = p[1];
	p0[2] = p[2];
	*anchor = 0;
	m_numPoints++;
}

void TetrahedronSystem::addTetrahedron(unsigned a, unsigned b, unsigned c, unsigned d)
{
	if(m_numTetrahedrons == m_maxNumTetrahedrons) return;
	unsigned *idx = &hostTretradhedronIndices()[m_numTetrahedrons * 4];
	idx[0] = a;
	idx[1] = b;
	idx[2] = c;
	idx[3] = d;
	m_numTetrahedrons++;
	
	addTriangle(idx[TetrahedronToTriangleVertexByFace[0][0]], 
	            idx[TetrahedronToTriangleVertexByFace[0][1]],
	            idx[TetrahedronToTriangleVertexByFace[0][2]]);
	
	addTriangle(idx[TetrahedronToTriangleVertexByFace[1][0]], 
	            idx[TetrahedronToTriangleVertexByFace[1][1]],
	            idx[TetrahedronToTriangleVertexByFace[1][2]]);
	
	addTriangle(idx[TetrahedronToTriangleVertexByFace[2][0]], 
	            idx[TetrahedronToTriangleVertexByFace[2][1]],
	            idx[TetrahedronToTriangleVertexByFace[2][2]]);
	
	addTriangle(idx[TetrahedronToTriangleVertexByFace[3][0]], 
	            idx[TetrahedronToTriangleVertexByFace[3][1]],
	            idx[TetrahedronToTriangleVertexByFace[3][2]]);
}

void TetrahedronSystem::addTriangle(unsigned a, unsigned b, unsigned c)
{
	if(m_numTriangles == m_maxNumTriangles) return;
	unsigned *idx = &hostTriangleIndices()[m_numTriangles * 3];
	idx[0] = a;
	idx[1] = b;
	idx[2] = c;
	m_numTriangles++;
}

const unsigned TetrahedronSystem::numTetrahedrons() const
{ return m_numTetrahedrons; }

const unsigned TetrahedronSystem::numPoints() const
{ return m_numPoints; }

const unsigned TetrahedronSystem::numTriangles() const
{ return m_numTriangles; }

const unsigned TetrahedronSystem::maxNumPoints() const
{ return m_maxNumPoints; }

const unsigned TetrahedronSystem::maxNumTetradedrons() const
{ return m_maxNumTetrahedrons; }

const unsigned TetrahedronSystem::numTriangleFaceVertices() const
{ return m_numTriangles * 3; }

float * TetrahedronSystem::hostX()
{ return (float *)m_hostX->data(); }

float * TetrahedronSystem::hostXi()
{ return (float *)m_hostXi->data(); }

float * TetrahedronSystem::hostV()
{ return (float *)m_hostV->data(); }

float * TetrahedronSystem::hostMass()
{ return (float *)m_hostMass->data(); }

unsigned * TetrahedronSystem::hostAnchor()
{ return (unsigned *)m_hostAnchor->data(); }

unsigned * TetrahedronSystem::hostTretradhedronIndices()
{ return (unsigned *)m_hostTetrahedronIndices->data(); }

unsigned * TetrahedronSystem::hostTriangleIndices()
{ return (unsigned *)m_hostTriangleIndices->data(); }

float TetrahedronSystem::totalInitialVolume()
{
	Vector3F * p = (Vector3F *)hostXi();
    unsigned * v = hostTretradhedronIndices();
    unsigned i;
	Vector3F t[4];
	unsigned a, b, c, d;
	float sum = 0.f;
	for(i=0; i<m_numTetrahedrons; i++) {
		a = v[0];
		b = v[1];
		c = v[2];
		d = v[3];
		t[0] = p[a];
		t[1] = p[b];
		t[2] = p[c];
		t[3] = p[d];
		sum += tetrahedronVolume(t);
	}
	return sum;
}

void TetrahedronSystem::calculateMass()
{
	float density = m_totalMass / totalInitialVolume();
	if(density < 0.f) density = -density;
	std::cout<<" density "<<density;
    const float base = 1.f/(float)m_numPoints;
    unsigned i;
    float * mass = hostMass();
    for(i=0; i< m_numPoints; i++) {
		if(isAnchoredPoint(i))
			mass[i] = 1e30f;
        else
			mass[i] = base;
    }
    
    Vector3F * p = (Vector3F *)hostXi();
    
    Vector3F v[4];
    unsigned a, b, c, d;
    unsigned *ind = hostTretradhedronIndices();
    float m;
    for(i=0; i<m_numTetrahedrons; i++) {
		a = ind[0];
		b = ind[1];
		c = ind[2];
		d = ind[3];
		
		v[0] = p[a];
		v[1] = p[b];
		v[2] = p[c];
		v[3] = p[d];
		
		m = density * tetrahedronVolume(v) * .25f;
		
		mass[a] += m;
		mass[b] += m;
		mass[c] += m;
		mass[d] += m;
		
		ind += 4;
	}
	/*
	for(i=0; i< m_numPoints; i++) {
	    std::cout<<" m "<<mass[i];
    }
    */
}

void TetrahedronSystem::setAnchoredPoint(unsigned i, unsigned anchorInd)
{
	unsigned * anchor = &hostAnchor()[i];
	*anchor = ((1<<30) | anchorInd);
}

bool TetrahedronSystem::isAnchoredPoint(unsigned i)
{ return (hostAnchor()[i] > (1<<29)); }

void TetrahedronSystem::getPointTetrahedronConnection(VicinityMap * vertTetConn)
{
	unsigned i;
	unsigned *ind = hostTretradhedronIndices();
	for(i=0; i<m_numTetrahedrons; i++) {
		vertTetConn[ind[0]][i] = 1;
		vertTetConn[ind[1]][i] = 1;
		vertTetConn[ind[2]][i] = 1;
		vertTetConn[ind[3]][i] = 1;
		ind += 4;
	}
}

void TetrahedronSystem::getTehrahedronTehrahedronConnectionL1(VicinityMap * tetTetConn, 
															VicinityMap * vertTetConn)
{
	unsigned i, j;
	unsigned *ind = hostTretradhedronIndices();
	for(i=0; i<m_numTetrahedrons; i++) {
		for(j=0; j<4; j++) {
			VicinityMapIter itvert = vertTetConn[ind[i*4 + j]].begin();
			for(; itvert != vertTetConn[ind[i*4 + j]].end(); ++itvert) {
				tetTetConn[itvert->first][i] = 1;
			}
		}
	}
}

void TetrahedronSystem::getTehrahedronTehrahedronConnectionL2(VicinityMap * dstConn, 
											VicinityMap * srcConn)
{
	
	unsigned i, j;
	for(i=0; i<m_numTetrahedrons; i++) {
		VicinityMapIter iti = srcConn[i].begin();
		for(; iti != srcConn[i].end(); ++iti) {
			j = iti->first;
			VicinityMapIter itj = srcConn[j].begin();
			for(; itj != srcConn[j].end(); ++itj) {
				dstConn[itj->first][i] = 1;
			}
		}
	}
}

void TetrahedronSystem::buildVicinityIndStart(VicinityMap * tetTetConn)
{
	m_hostTetrahedronVicinityStart->create((m_numTetrahedrons+1)*4);
	unsigned * tvstart = (unsigned *)m_hostTetrahedronVicinityStart->data();
	
	std::cout<<" n tet "<<m_numTetrahedrons;
	
	unsigned count = 0;
	unsigned i;
	for(i=0; i<m_numTetrahedrons; i++) {
		tvstart[i] = count;
		count += tetTetConn[i].size();
	}
	tvstart[m_numTetrahedrons] = count;

#if PRINT_VICINITY
	std::cout<<" vicinity size "<<tvstart[m_numTetrahedrons];
#endif
	
	m_tetrahedronVicinitySize = count;
	
	m_hostTetrahedronVicinityInd->create(count * 4);
	unsigned * tvind = (unsigned *)m_hostTetrahedronVicinityInd->data();
	
	if(m_tetrahedronVicinitySize == m_numTetrahedrons) {
#if PRINT_VICINITY
		std::cout<<" no tetrahedrons are connected to each other";
#endif
		for(i=0; i<m_numTetrahedrons; i++) {
			tvind[i] = i;
		}
		return;
	}
	
	count = 0;
	for(i=0; i<m_numTetrahedrons; i++) {
	
#if PRINT_VICINITY
		std::cout<<"\n t"<<i<<" ["<<tvstart[i]<<":] ";
#endif

		VicinityMapIter ittet = tetTetConn[i].begin();
		for(; ittet != tetTetConn[i].end(); ++ittet) {
			
#if PRINT_VICINITY
			std::cout<<" "<<ittet->first;
#endif
			tvind[count] = ittet->first;
			count++;
		}
	}
}

void TetrahedronSystem::createL1Vicinity()
{
#if PRINT_VICINITY
	std::cout<<" create L1 vicinity\n";
#endif

	VicinityMap * vertTetConn = new VicinityMap[m_numPoints];
	getPointTetrahedronConnection(vertTetConn);
	
	VicinityMap * tetTetConn = new VicinityMap[m_numTetrahedrons];
	getTehrahedronTehrahedronConnectionL1(tetTetConn, vertTetConn);
	
	buildVicinityIndStart(tetTetConn);
	
	unsigned i;
	for(i=0; i<m_numPoints; i++) vertTetConn[i].clear();
	delete[] vertTetConn;
	
	for(i=0; i<m_numTetrahedrons; i++) tetTetConn[i].clear();
	delete[] tetTetConn;
}

void TetrahedronSystem::createL2Vicinity()
{
#if PRINT_VICINITY
	std::cout<<" create L2 vicinity\n";
#endif

	VicinityMap * vertTetConn = new VicinityMap[m_numPoints];
	getPointTetrahedronConnection(vertTetConn);	
	
	VicinityMap * tetTetConn1 = new VicinityMap[m_numTetrahedrons];
	getTehrahedronTehrahedronConnectionL1(tetTetConn1, vertTetConn);
	
	VicinityMap * tetTetConn2 = new VicinityMap[m_numTetrahedrons];
	getTehrahedronTehrahedronConnectionL2(tetTetConn2, tetTetConn1);
	
	buildVicinityIndStart(tetTetConn2);
	
	unsigned i;
	for(i=0; i<m_numPoints; i++) vertTetConn[i].clear();
	delete[] vertTetConn;
	
	for(i=0; i<m_numTetrahedrons; i++) tetTetConn1[i].clear();
	delete[] tetTetConn1;
	
	for(i=0; i<m_numTetrahedrons; i++) tetTetConn2[i].clear();
	delete[] tetTetConn2;
}

const unsigned TetrahedronSystem::numTetrahedronVicinityInd() const
{ return m_tetrahedronVicinitySize; }

unsigned * TetrahedronSystem::hostTetrahedronVicinityInd()
{ (unsigned *)m_hostTetrahedronVicinityInd->data(); }

unsigned * TetrahedronSystem::hostTetrahedronVicinityStart()
{ (unsigned *)m_hostTetrahedronVicinityStart->data(); }
//~:
