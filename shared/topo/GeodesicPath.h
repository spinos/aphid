/*
 *  GeodesicPath.h
 *
 *  one path to each tip, length is distance from root to that tip
 *  for each vertex, calculate geodesic distance to root and each tip
 *  select the path if sum of distance to root and distance to tip is close to path length 
 *  level set distance to root to build curve
 *  
 *
 *  Created by jian zhang on 10/26/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TOPO_GEODESIC_PATH_H
#define APH_TOPO_GEODESIC_PATH_H

#include <math/Vector3F.h>
#include <boost/scoped_array.hpp>
#include <deque>
#include <map>

namespace aphid {

namespace topo {

class PathData {

	int _nv;
	
	struct SetMean {
		Vector3F _pos;
		int _nv;
	};
	
	std::map<int, SetMean > m_groups;
	
public:
	PathData();
	
	bool isEmpty() const;
	
	void addToSet(int iset, const Vector3F& pos);
	
	int numSets();
	void getSetMeans(Vector3F* dest);
	
};

class GeodesicPath {

	int m_numVertices;
	
public:
	GeodesicPath();
	virtual ~GeodesicPath();
	
	void create(int nv);
	void findPathToTip();
	void colorByDistanceToRoot(const float& maxD);
	void clearAllPath();
	bool build(const float& unitD,
			const Vector3F* pos);
	
	const std::deque<int>& rootNodeIndices() const;
	const std::deque<int>& tipNodeIndices() const;
	
	void addRoot(int x);
	void addTip(int x);
	void setLastRootNodeIndex(int x);
	void setLastTipNodeIndex(int x);
	
	bool hasRoot() const;
	bool hasTip() const;
	int numRoots() const;
	int numTips() const;
	const int& numJoints() const;
	const int& numVertices() const;
	const float* dysCols() const;
	float* distanceToRoot();
	float* distanceToLastTip();
	int lastTipNodeIndex();
	const Vector3F* jointPos() const;
	
	const float* dspRootColR() const;
	const float* dspTipColR(int i) const;
	
protected:
	static const float DspRootColor[3];
	static const float DspTipColor[8][3];
	
	bool buildSkeleton(PathData* pds);
	
private:

	std::deque<int> m_rootNodeIndices;
	std::deque<int> m_tipIndices;
	
typedef boost::scoped_array<int> IntArrTyp;
	IntArrTyp m_pathInd;
	
typedef boost::scoped_array<float> FltArrTyp;
	FltArrTyp m_dysCols;
	FltArrTyp m_dist2Root;
/// per path
	std::deque<FltArrTyp* > m_dist2Tip;
	FltArrTyp m_distDiff;
	
	int m_numJoints;
	IntArrTyp m_jointCounts;
	IntArrTyp m_jointBegins;

typedef boost::scoped_array<Vector3F> PntArrTyp;
	PntArrTyp m_jointPos;
	
};

}

}

#endif
