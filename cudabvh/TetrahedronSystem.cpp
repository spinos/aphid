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
#include <ATetrahedronMesh.h>
#include <HTetrahedronMesh.h>

#define PRINT_VICINITY 0
#define GRDW 39
#define GRDH 39
#define NTET 1600
#define NPNT 6400

float TetrahedronSystem::Density = 100.f;

TetrahedronSystem::TetrahedronSystem() 
{
	m_hostTetrahedronVicinityInd = new BaseBuffer;
	m_hostTetrahedronVicinityStart = new BaseBuffer;
	m_hostElementValue = new BaseBuffer;
	create(NTET, NTET * 4, NPNT);
	m_hostElementValue->create(NTET * 4);
    
	float * hv = &hostV()[0];
	
	unsigned i, j;
	float vy = 3.95f;
	float vrx, vry, vrz, vr, vs;
	for(j=0; j < GRDH; j++) {
		for(i=0; i<GRDW; i++) {
		    vs = 1.75f + RandomF01() * 1.5f;
			Vector3F base(9.3f * i, 9.3f * j, 0.f * j);
			Vector3F right = base + Vector3F(1.75f, 0.f, 0.f) * vs;
			Vector3F front = base + Vector3F(0.f, 0.f, 1.75f) * vs;
			Vector3F top = base + Vector3F(0.f, 1.75f, 0.f) * vs;
			if((j&1)==0) {
			    right.y = top.y-.1f;
			}
			else {
			    base.x -= .085f * vs;
			}
			
			vrx = 0.725f * (RandomF01() - .5f);
			vry = 1.f  * (RandomF01() + 1.f)  * vy;
			vrz = 0.732f * (RandomF01() - .5f);
			vr = 0.13f * RandomF01();
			
			addPoint(&base.x);
			hv[0] = vrx + vr;
			hv[1] = vry;
			hv[2] = vrz - vr;
			hv+=3;
			addPoint(&right.x);
			hv[0] = vrx - vr;
			hv[1] = vry;
			hv[2] = vrz + vr;
			hv+=3;
			addPoint(&top.x);
			hv[0] = vrx + vr;
			hv[1] = vry;
			hv[2] = vrz + vr;
			hv+=3;
			addPoint(&front.x);
			hv[0] = vrx - vr;
			hv[1] = vry;
			hv[2] = vrz - vr;
			hv+=3;

			unsigned b = (j * GRDW + i) * 4;
			addTetrahedron(b, b+1, b+2, b+3);		
		}
		vy = -vy;
	}
	setTotalMass(100.f);
    calculateMass();
    createL2Vicinity();
}

TetrahedronSystem::TetrahedronSystem(ATetrahedronMesh * md)
{
	m_hostTetrahedronVicinityInd = new BaseBuffer;
	m_hostTetrahedronVicinityStart = new BaseBuffer;
	m_hostElementValue = new BaseBuffer;
	create(md->numTetrahedrons() + 100, md->numTetrahedrons() * 4 + 400, md->numPoints() + 400);
	m_hostElementValue->create((md->numTetrahedrons() + 100) * 4);
    
    Vector3F * p = md->points();
	unsigned i;
	for(i=0; i< md->numPoints(); i++)
	    addPoint((float *)&p[i]);
	
	unsigned * ind = (unsigned *)md->indices();
	for(i=0; i< md->numTetrahedrons(); i++)
		addTetrahedron(ind[i*4], ind[i*4+1], ind[i*4+2], ind[i*4+3]);
		
	unsigned * anchor = md->anchors();
	for(i=0; i< md->numPoints(); i++) {
        //std::cout<<"a "<<anchor[i];
		if(anchor[i] > 0) hostAnchor()[i] = anchor[i];
	}
// density of nylon = 1.15 g/cm^3
// very low density is unstable
    std::cout<<"\n initial volume "<<md->volume()
       <<"\n initial mass "<<(100.f * md->volume())
       <<"\n";
    setTotalMass(100.f * md->volume());
    calculateMass();
    createL2Vicinity();
}

TetrahedronSystem::~TetrahedronSystem() 
{
	delete m_hostTetrahedronVicinityInd;
	delete m_hostTetrahedronVicinityStart;
}

float TetrahedronSystem::totalInitialVolume()
{
	const unsigned n = numTetrahedrons();
	Vector3F * p = (Vector3F *)hostXi();
    unsigned * v = hostTetrahedronIndices();
    unsigned i;
	Vector3F t[4];
	unsigned a, b, c, d;
	float sum = 0.f;
	for(i=0; i<n; i++) {
		a = v[0];
		b = v[1];
		c = v[2];
		d = v[3];
		t[0] = p[a];
		t[1] = p[b];
		t[2] = p[c];
		t[3] = p[d];
		sum += tetrahedronVolume(t);
        v+= 4;
	}
	return sum;
}

void TetrahedronSystem::calculateMass()
{
	const unsigned np = numPoints();
	const unsigned nt = numTetrahedrons();
	const float density = totalMass() / totalInitialVolume();
    const float base = 1.f/(float)np;
    unsigned i;
    float * mass = hostMass();
    for(i=0; i< np; i++) {
		if(isAnchoredPoint(i))
			mass[i] = 1e20f;
        else
			mass[i] = base;
    }
    
    Vector3F * p = (Vector3F *)hostXi();
    
    Vector3F v[4];
    unsigned a, b, c, d;
    unsigned *ind = hostTetrahedronIndices();
    float m;
    for(i=0; i<nt; i++) {
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

void TetrahedronSystem::getPointTetrahedronConnection(VicinityMap * vertTetConn)
{
	unsigned i;
	unsigned *ind = hostTetrahedronIndices();
	const unsigned nt = numTetrahedrons();
	for(i=0; i<nt; i++) {
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
	unsigned *ind = hostTetrahedronIndices();
	const unsigned nt = numTetrahedrons();
	for(i=0; i<nt; i++) {
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
	const unsigned nt = numTetrahedrons();
	unsigned i, j;
	for(i=0; i<nt; i++) {
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
	const unsigned nt = numTetrahedrons();
	m_hostTetrahedronVicinityStart->create((nt+1)*4);
	unsigned * tvstart = (unsigned *)m_hostTetrahedronVicinityStart->data();
	
	std::cout<<" n tet "<<nt;
	
	unsigned maxConn = 0;
	unsigned minConn = 1000;
	
	unsigned count = 0;
	unsigned i;
	for(i=0; i<nt; i++) {
		tvstart[i] = count;
		count += tetTetConn[i].size();
		
		if(maxConn < tetTetConn[i].size()) maxConn = tetTetConn[i].size();
		if(minConn > tetTetConn[i].size()) minConn = tetTetConn[i].size();
	}
	tvstart[nt] = count;
	
	std::cout<<" min/max n connections "<<minConn<<"/"<<maxConn<<"\n";

#if PRINT_VICINITY
	std::cout<<" vicinity size "<<tvstart[nt];
#endif
	
	m_tetrahedronVicinitySize = count;
	
	m_hostTetrahedronVicinityInd->create(count * 4);
	unsigned * tvind = (unsigned *)m_hostTetrahedronVicinityInd->data();
	
	if(m_tetrahedronVicinitySize == nt) {
#if PRINT_VICINITY
		std::cout<<" no tetrahedrons are connected to each other";
#endif
		for(i=0; i<nt; i++) {
			tvind[i] = i;
		}
		return;
	}
	
	count = 0;
	for(i=0; i<nt; i++) {
	
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
	const unsigned nt = numTetrahedrons();
	const unsigned np = numPoints();

	VicinityMap * vertTetConn = new VicinityMap[np];
	getPointTetrahedronConnection(vertTetConn);
	
	VicinityMap * tetTetConn = new VicinityMap[nt];
	getTehrahedronTehrahedronConnectionL1(tetTetConn, vertTetConn);
	
	buildVicinityIndStart(tetTetConn);
	
	unsigned i;
	for(i=0; i<np; i++) vertTetConn[i].clear();
	delete[] vertTetConn;
	
	for(i=0; i<nt; i++) tetTetConn[i].clear();
	delete[] tetTetConn;
}

void TetrahedronSystem::createL2Vicinity()
{
#if PRINT_VICINITY
	std::cout<<" create L2 vicinity\n";
#endif
	const unsigned nt = numTetrahedrons();
	const unsigned np = numPoints();

	VicinityMap * vertTetConn = new VicinityMap[np];
	getPointTetrahedronConnection(vertTetConn);	
	
	VicinityMap * tetTetConn1 = new VicinityMap[nt];
	getTehrahedronTehrahedronConnectionL1(tetTetConn1, vertTetConn);
	
	VicinityMap * tetTetConn2 = new VicinityMap[nt];
	getTehrahedronTehrahedronConnectionL2(tetTetConn2, tetTetConn1);
	
	buildVicinityIndStart(tetTetConn2);
	
	unsigned i;
	for(i=0; i<np; i++) vertTetConn[i].clear();
	delete[] vertTetConn;
	
	for(i=0; i<nt; i++) tetTetConn1[i].clear();
	delete[] tetTetConn1;
	
	for(i=0; i<nt; i++) tetTetConn2[i].clear();
	delete[] tetTetConn2;
}

const unsigned TetrahedronSystem::numTetrahedronVicinityInd() const
{ return m_tetrahedronVicinitySize; }

unsigned * TetrahedronSystem::hostTetrahedronVicinityInd()
{ return (unsigned *)m_hostTetrahedronVicinityInd->data(); }

unsigned * TetrahedronSystem::hostTetrahedronVicinityStart()
{ return (unsigned *)m_hostTetrahedronVicinityStart->data(); }

const int TetrahedronSystem::elementRank() const
{ return 4; }

const unsigned TetrahedronSystem::numElements() const
{ return numTetrahedrons(); }

float * TetrahedronSystem::hostElementValue() const
{ return (float *)m_hostElementValue->data(); }
//~:
