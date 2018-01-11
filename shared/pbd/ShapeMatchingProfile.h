/*
 *  ShapeMatchingProfile.h
 *  
 *
 *  Created by jian zhang on 1/14/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_PBD_SHAPE_MATCHING_PROFILE_H
#define APH_PBD_SHAPE_MATCHING_PROFILE_H

#include <math/Vector3F.h>
#include <boost/scoped_array.hpp>

namespace aphid {
namespace pbd {

struct RegionVE;

class ShapeMatchingProfile {

/// position_0
	boost::scoped_array<Vector3F > m_x0;
/// inverse mass
	boost::scoped_array<float > m_invmass;
	boost::scoped_array<int > m_regionVertexBegin;
	boost::scoped_array<int > m_regionEdgeBegin;
/// region varying
	boost::scoped_array<int > m_vertices;	
/// v1 and v2 per edge region varying
	boost::scoped_array<int > m_edges;	
	int m_numPoints;
	int m_numRegions;
	
public:
	ShapeMatchingProfile();
	
	const int& numPoints() const;
	const int& numRegions() const;
	const Vector3F* x0() const;
	const float* inverseMass() const;
/// i-th region vertices and edges
	void getRegionVE(RegionVE& ve, const int& i) const;
	
/// each strand has 
/// 2n vertices
/// n-1 segments
/// n-2 regions
/// - v0 - v3 -
///   |    |
/// - v1 - v2 -	
/// one strand test
	void createTestStrand();

protected:

};

}
}

#endif
