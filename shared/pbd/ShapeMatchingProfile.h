/*
 *  ShapeMatchingProfile.h
 *  
 * each strand has 
 * 2n vertices
 * n-1 segments
 * n-2 regions
 * each region has 6 vertices 7 edges
 * v0 - v3 - v4
 * |    |    |
 * v1 - v2 - v5
 *  Created by jian zhang on 1/14/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_PBD_SHAPE_MATCHING_PROFILE_H
#define APH_PBD_SHAPE_MATCHING_PROFILE_H

#include <math/Vector3F.h>
#include <boost/scoped_array.hpp>
#include <vector>

namespace aphid {
namespace pbd {

struct RegionVE;

struct StrandParam {

	Vector3F _binormal;
/// root mass
	float _mass0;
/// tip mass
	float _mass1;
	float _stiffness0;
	float _stiffness1;
	int _padding;
	
};

class ShapeMatchingProfile {

	std::vector<Vector3F > m_strandX0;
	std::vector<StrandParam > m_strandParams;
	std::vector<int > m_strandBegin;
	
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
	float m_avgSegLen;
	float m_lod;
	int m_numPoints;
	int m_numRegions;
	
public:

	ShapeMatchingProfile();
	
	void setLod(const float& x);
	
	void clearStrands();
	void addStrandPoint(const Vector3F& x0);
	void finishStrand(const StrandParam& param);
	void buildProfile();
	
	const int& numPoints() const;
	const int& numRegions() const;
	const float& averageSegmentLength() const;
	float detailSize() const;
	const Vector3F* x0() const;
	const float* inverseMass() const;
/// i-th region vertices and edges
	void getRegionVE(RegionVE& ve, const int& i) const;

protected:

private:
	
	void addSegLen(float& segLenSum, int& segCount, const int& istrand);
	void buildStrand(const int& istrand);
	void countRegions(int& regionCount, const int& i);
	void countRegionEdges(int& regionCount, int& edgeCount, int& vertexCount,
				const int& istrand);
	void buildRegions(int& regionCount, const int& istrand);
/// i-th strand
	int strandNumPoints(const int& i) const;

};

}
}

#endif
