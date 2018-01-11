/*
 *  ShapeMatchingContext.cpp
 *  
 *
 *  Created by jian zhang on 1/10/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "ShapeMatchingContext.h"
#include <geom/ParallelTransport.h>
#include <math/Matrix33F.h>
#include <sdb/Sequence.h>
#include <math/miscfuncs.h>
#include <math/sparseLinearMath.h>

namespace aphid {
namespace pbd {

ShapeMatchingContext::ShapeMatchingContext() :
m_numStrands(0),
m_numEdges(0)
{}

ShapeMatchingContext::~ShapeMatchingContext()
{}

void ShapeMatchingContext::create()
{
/// one strand test
	Vector3F g[32];
	for(int i=0;i<32;++i) {
		g[i].set(10.f + (.5f + .005 * i) * i, 10.f - .05f * i + 1.2f * (.5f + .005f * i) * sin(0.5f * i), 1.3f * (.5f + .005f * i) * cos(.5f * i) );
	}
	
	m_numStrands = 1;
	m_strandBegin = new int[m_numStrands + 1];
	m_strandBegin[0] = 0;
	m_strandBegin[1] = 64;
	
	const int np = 64;
	const int hnp = 32;
	const int nr = 30;
	pbd::ParticleData* part = particles();
	part->createNParticles(np);
	
	Vector3F p0p1 = g[1] - g[0];
	Matrix33F frm;
	ParallelTransport::FirstFrame(frm, p0p1, Vector3F(0.f, 1.f, 1.f) );
	Vector3F nml = ParallelTransport::FrameUp(frm);
	
	part->setParticle(g[0] - nml * .25f, 0);
	part->setParticle(g[0] + nml * .25f, 1);
	
	static const int hilb[2][2] = {{1, 0}, {0, 1}};
	
	Vector3F p1p2;
	for(int i=1;i<hnp;++i) {
		p1p2 = g[i+1] - g[i];
		ParallelTransport::RotateFrame(frm, p0p1, p1p2);
		nml = ParallelTransport::FrameUp(frm);
		
		part->setParticle(g[i] + nml * .25f, i * 2 + hilb[i&1][0]);
		part->setParticle(g[i] - nml * .25f, i * 2 + hilb[i&1][1]);
		
		p0p1 = p1p2;
	}
	
	for(int i=0;i<np;++i) {
		part->invMass()[i] = 2.f;
	}
/// lock first segment
	for(int i=0;i<4;++i) {
		part->invMass()[i] = 0.f;
	}
/// map edges
	sdb::Sequence<sdb::Coord2> edgeMap;
	edgeMap.insert(sdb::Coord2(0, 1));
	
	int prev[2] = {1, 0};
	for(int i=1;i<hnp;++i) {
/// prev1 - v11
///	        |
/// prev0 - v10
		int v10 = i * 2 + hilb[i&1][0];
		int v11 = i * 2 + hilb[i&1][1];
		
		edgeMap.insert(sdb::Coord2(v10, v11).ordered() );
		edgeMap.insert(sdb::Coord2(v10, prev[0]).ordered() );
		edgeMap.insert(sdb::Coord2(v11, prev[1]).ordered() );
		
		prev[0] = v10;
		prev[1] = v11;
	}
	
	const int ne = edgeMap.size();
	
	m_edgeInds = new sdb::Coord2[ne];
	int ie = 0;
	edgeMap.begin();
	while(!edgeMap.end() ) {
		m_edgeInds[ie] = edgeMap.key();
		ie++;
		edgeMap.next();
	}
	m_numEdges = ie;
	
/// rest edge lengths
	m_restEdgeLs = new float[m_numEdges];
	
	const Vector3F* vs = part->pos();
	for(int i=0;i<m_numEdges;++i) {
		m_restEdgeLs[i] = (vs[m_edgeInds[i].x]).distanceTo(vs[m_edgeInds[i].y]);
	}

	int dim = np;
/// t <- 1/30 sec
	const float oneovert2 = 900.f;
	m_lhsMat = new SparseMatrix<float>;
	m_lhsMat->create(dim, dim);

	const float attachmentStiffness = 75;
	const float stretchStiffness = 20;
	const float fixed = 100000;
	const float springK = 300000;
	for(int i=0;i<np;++i) {
/// for each particle add mass diagonal
/// (M + fixed) / t^2
		const float& imi = part->invMass()[i];
		if(imi > 0.f) {
			m_lhsMat->set(i, i, oneovert2 / imi);
		} else {
			m_lhsMat->set(i, i, oneovert2 * fixed);
		}
	}
	
	for(int i=0;i<m_numEdges;++i) {
/// for each edge
		const sdb::Coord2& ei = m_edgeInds[i];
/// +k center
		m_lhsMat->add(ei.x, ei.x, springK);
		m_lhsMat->add(ei.y, ei.y, springK);
/// -k neighbor
		m_lhsMat->add(ei.x, ei.y, -springK);
		m_lhsMat->add(ei.y, ei.x, -springK);
	}
	
	m_lhsMat->printMatrix();
	
}

const int& ShapeMatchingContext::numEdges() const
{ return m_numEdges; }

void ShapeMatchingContext::getEdge(int& v1, int& v2, const int& i) const
{ 
	v1 = m_edgeInds[i].x;
	v2 = m_edgeInds[i].y; 
}

}
}