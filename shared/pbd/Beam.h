/*
 *  Beam.h
 *  pbd
 *
 *  n segment n+1 particles n ghost_particles
 *  placed on 3-piece hermite curve
 *
 *  Created by jian zhang on 7/29/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_PBD_BEAM_H
#define APH_PBD_BEAM_H

#include <math/HermiteInterpolatePiecewise.h>
#include <math/Vector3F.h>
#include <boost/scoped_array.hpp>

namespace aphid {

class MatrixC33F;

namespace pbd {

class Beam : public HermiteInterpolatePiecewise<float, Vector3F > {

	boost::scoped_array<Vector3F > m_p;
	boost::scoped_array<Vector3F > m_gp;
/// order to enforce constraints using
/// bidirectional interleaving permutation
/// 1  3 5 7 9
///  10 8 6 4 2
	boost::scoped_array<int > m_constraintSegInd;
/// ref for 1st seg ghost
	Vector3F m_ref;
	int m_numSegs;
	
public:
	Beam();
	virtual ~Beam();
/// before create segs
/// let material frame x close to each other
	void setGhostRef(const Vector3F& ref);
/// after set hermite control points and tangets
/// spp is num seg per piece	
	void createNumSegments(int spp);
/// compute particle and ghost pnts  	
	void calculatePnts();
	
	const int& numSegments() const;
	const int numParticles() const;
	const int& numGhostParticles() const;
	
	const Vector3F& getParticlePnt(int i) const;
	const Vector3F& getGhostParticlePnt(int i) const;
	const int& getConstraintSegInd(int i) const;
	Vector3F getSegmentMidPnt(int i) const;
	MatrixC33F getMaterialFrame(int i) const;
	
private:
	void permutateConstraintInd();
	
};

}

}

#endif
