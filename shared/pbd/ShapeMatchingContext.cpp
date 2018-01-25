/*
 *  ShapeMatchingContext.cpp
 *  
 *
 *  Created by jian zhang on 1/10/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "ShapeMatchingContext.h"
#include <math/ConjugateGradient.h>
#include "ShapeMatchingProfile.h"

namespace aphid {
namespace pbd {

ShapeMatchingContext::ShapeMatchingContext() :
m_lhsMat(0),
m_cg(0),
m_stiffness(500.f)
{}

ShapeMatchingContext::~ShapeMatchingContext()
{ clearConstraints(); }

void ShapeMatchingContext::clearConstraints()
{
	std::vector<ShapeMatchingRegion* >::iterator it = m_regions.begin();
	for(;it!=m_regions.end();++it) {
		delete *it;
	}
	m_regions.end();
	if(m_cg)
		delete m_cg;
	if(m_lhsMat)
		delete m_lhsMat;
}

void ShapeMatchingContext::create(const ShapeMatchingProfile& prof)
{	
	resetCollisionGrid(prof.detailSize() );
	clearConstraints();
	
	const int& np = prof.numPoints();
	
	pbd::ParticleData* part = particles();
	part->createNParticles(np);
	
	const Vector3F* p0 = prof.x0();
	for(int i=0;i<np;++i) {
		part->setParticle(p0[i], i);
	}
	
	for(int i=0;i<np;++i) {
		part->invMass()[i] = prof.inverseMass()[i];
	}

/// add all regions
	RegionVE ve;
	const int& nr = prof.numRegions();
	std::cout<<"\n ShapeMatchingContext::create n regions "<<nr;
	for(int i=0;i<nr;++i) {
		prof.getRegionVE(ve, i);
		
		ShapeMatchingRegion* ri = new ShapeMatchingRegion;
		ri->createRegion(ve, (const float* )prof.x0(), prof.inverseMass() );
		m_regions.push_back(ri);
	}

	const int dim = np * 3;
/// t <- 1/30 sec
	static const float oneovert2 = 900.f;
	m_lhsMat = new SparseMatrix<float>;
/// row-major
	m_lhsMat->create(dim, dim, true);

	float movert2;
	for(int i=0;i<np;++i) {
/// for each particle add mass diagonal
/// (M + fixed) / t^2
		movert2 = part->getMass(i) * oneovert2;
		
		m_lhsMat->set(i * 3, i * 3, movert2);
		m_lhsMat->set(i * 3 + 1, i * 3 + 1, movert2);
		m_lhsMat->set(i * 3 + 2, i * 3 + 2, movert2);
	}

	int v1, v2;
	for(int i=0;i<nr;++i) {
/// for each region
		const ShapeMatchingRegion* ri = m_regions[i];
		const int& ne = ri->numEdges();
		for(int j=0;j<ne;++j) {
			ri->getEdge(v1, v2, j); 
/// +k center
		m_lhsMat->add(v1 * 3    , v1 * 3    , m_stiffness);
		m_lhsMat->add(v1 * 3 + 1, v1 * 3 + 1, m_stiffness);
		m_lhsMat->add(v1 * 3 + 2, v1 * 3 + 2, m_stiffness);
		m_lhsMat->add(v2 * 3    , v2 * 3    , m_stiffness);
		m_lhsMat->add(v2 * 3 + 1, v2 * 3 + 1, m_stiffness);
		m_lhsMat->add(v2 * 3 + 2, v2 * 3 + 2, m_stiffness);
/// -k neighbor
		m_lhsMat->add(v1 * 3    , v2 * 3    , -m_stiffness);
		m_lhsMat->add(v1 * 3 + 1, v2 * 3 + 1, -m_stiffness);
		m_lhsMat->add(v1 * 3 + 2, v2 * 3 + 2, -m_stiffness);
		m_lhsMat->add(v2 * 3    , v1 * 3    , -m_stiffness);
		m_lhsMat->add(v2 * 3 + 1, v1 * 3 + 1, -m_stiffness);
		m_lhsMat->add(v2 * 3 + 2, v1 * 3 + 2, -m_stiffness);
		}
	}
	
	//m_lhsMat->printMatrix();
	m_cg = new ConjugateGradient<float>(m_lhsMat);
	
}

void ShapeMatchingContext::setStiffness(const float& x)
{ m_stiffness = x; }

int ShapeMatchingContext::numRegions() const
{ return m_regions.size(); }

const ShapeMatchingRegion* ShapeMatchingContext::region(const int& i) const
{ return m_regions[i]; }

void ShapeMatchingContext::applyPositionConstraint()
{
	pbd::ParticleData* part = particles();
	const int& np = part->numParticles();
	const int dim = np * 3;
	const int nr = numRegions();
	
	DenseVector<float> q_n1;
	q_n1.create(dim);
	q_n1.copyData((const float*)part->projectedPos() );
/// rhs	
	DenseVector<float> s_n; 
	s_n.copy(q_n1);
	
	static const float oneovert2 = 900.f;
	float movert2;
	for(int i=0;i<np;++i) {
		movert2 = part->getMass(i) * oneovert2;
		
/// M / h^2 S_n		
		s_n[i * 3] *= movert2;
		s_n[i * 3 + 1] *= movert2;
		s_n[i * 3 + 2] *= movert2;
	}
	
	float pSpring[6];
	int v1, v2, g1, g2;
	DenseVector<float> b;
	
	for(int k=0;k<2;++k) {
	
		b.copy(s_n);
	
/// + sigma (w_i S_i^T A_i^T B_i p)
		for(int i=0;i<nr;++i) {
/// for each region
			ShapeMatchingRegion* ri = m_regions[i];
/// center only?		
			ri->updateRegion(q_n1.v() );
		
			const int& ne = ri->numEdges();
			for(int j=0;j<ne;++j) {
				ri->getEdge(v1, v2, j);
				ri->getGoalInd(g1, g2, j);
				
				ri->solvePositionConstraint(pSpring, q_n1.v(), v1, g1);
				ri->solvePositionConstraint(&pSpring[3], q_n1.v(), v2, g2);
			
				b[v1 * 3    ] += m_stiffness * (pSpring[0] - pSpring[3]);
				b[v1 * 3 + 1] += m_stiffness * (pSpring[1] - pSpring[4]);
				b[v1 * 3 + 2] += m_stiffness * (pSpring[2] - pSpring[5]);
				
				b[v2 * 3    ] += m_stiffness * (-pSpring[0] + pSpring[3]);
				b[v2 * 3 + 1] += m_stiffness * (-pSpring[1] + pSpring[4]);
				b[v2 * 3 + 2] += m_stiffness * (-pSpring[2] + pSpring[5]);
			}
		}

		m_cg->solve(q_n1, b);
	
	}
	
	q_n1.extractData((float*)part->projectedPos());
}

}
}