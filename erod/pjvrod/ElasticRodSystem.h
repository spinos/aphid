/*
 *  ElasticRodSystem.h
 *
 *  Created by jian zhang on 1/6/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef APH_ELASTIC_ROD_SYSTEM_H
#define APH_ELASTIC_ROD_SYSTEM_H

#include <math/Vector3F.h>
#include <math/Vector2F.h>
#include <boost/scoped_array.hpp>

namespace aphid {
namespace pbd {

class ElasticRodSystem {

/// x_i
	boost::scoped_array<Vector3F > m_pos;
/// e_i
	boost::scoped_array<Vector3F > m_edges;
/// curvature binormal
	boost::scoped_array<Vector3F > m_kb;
/// axes of the cross section
	boost::scoped_array<Vector3F > m_m1;
	boost::scoped_array<Vector3F > m_m2;
/// |e_i|
	boost::scoped_array<float > m_restEdgeL;
/// l_i <- |e_i-1| + |e_i|
	boost::scoped_array<float > m_restRegionL;
/// material curvature
	boost::scoped_array<Vector2F > m_restWprev;
	boost::scoped_array<Vector2F > m_restWnext;
/// m1 of e_0
	Vector3F m_u0;
	int m_numSegments;
	
public:
	ElasticRodSystem();
	virtual ~ElasticRodSystem();
	
	void create();
	
	int numNodes() const;
	const int& numSegments() const;
	const Vector3F* positions() const;
/// i-th segment
	void getBishopFrame(Vector3F& c,
				Vector3F& t, Vector3F& u, Vector3F& v,
				int i) const;
	
private:

	void computeEdges();
	void computeLengths();
/// kb_i <- 2 (e_i-1 x e_i) / (|e_i-1||e_i| + e_i-1 . e_i)
	void computeCurvatureBinormals();
	void computeBishopFrame();
	void extractSinAndCos(float& sinPhi, float& cosPhi,
			const float& kdk) const;
	void computeMaterialCurvature();
	void computeW(Vector2F& dst, const Vector3F& kb, 
			const Vector3F& m1, const Vector3F& m2) const;
	void updateCurrentState();
			
};

}
}

#endif


