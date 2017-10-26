/*
 *  DistancePath.h
 *
 *  one path to each tip, length is distance from root to that tip
 *  for each vertex, calculate geodesic distance to root and each tip
 *  select the path if sum of distance to root and distance to tip is close to path length 
 *  level set distance to root to build joints
 *
 *  Created by jian zhang on 10/26/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TOPO_DISTANCE_PATH_H
#define APH_TOPO_DISTANCE_PATH_H

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
		float _val;
		int _nv;
	};
	
	std::map<int, SetMean > m_groups;
	
public:
	PathData();
	
	void addToSet(int iset, const Vector3F& pos,
					const float& val);
	
	int numSets();
	void average();
	void getSetPos(Vector3F* dest);
	int getClosestSet(const float& val);
	
};

class DistancePath {

	int m_numVertices;
	
public:
	DistancePath();
	virtual ~DistancePath();
	
	void create(int nv);
	void findPathToTip();
	void colorByRegionLabels();
	void colorByDistanceToRoot(const float& maxD);
	void clearAllPath();
	
	void setRootNodeIndex(int x);
	void addSeed(int x);
	void setLastRootNodeIndex(int x);
	void setLastTipNodeIndex(int x);
	
	int numRegions() const;
	int numSeeds() const;
	const int& numVertices() const;
	const float* dysCols() const;
	float* distanceToRoot();
	int lastTipNodeIndex();
	float* distanceToSite(int i);
	float* distanceToSeed(int i);
	
	const int& siteNodeIndex(int i) const;
	const int& rootNodeInd() const;
	const int& seedNodeIndex(int i) const;
	
	const float* dspRegionColR(int i) const;
	const int* vertexPathInd() const;
	const int* vertexSetInd() const;
	
	int* vertexLabels();
	
	static const float DspRegionColor[8][3];

protected:
	
/// assign label to known initial sites
/// unknown sites <- -1
	void labelRootAndSeedPoints();
	
	bool buildLevelSet(PathData* dest,
			const float& unitD,
			const Vector3F* pos);
	
private:
	void addSite(int x);
	void setVertexSetInd(PathData* dest);
	
private:
/// first is to root
	std::deque<int> m_siteIndices;
	std::deque<float> m_pathLength;
	
typedef boost::scoped_array<int> IntArrTyp;
	IntArrTyp m_pathLab;
/// path varying
	IntArrTyp m_setInd;
	
typedef boost::scoped_array<float> FltArrTyp;
/// per vertex
	FltArrTyp m_dysCols;
/// first is to root
	std::deque<FltArrTyp* > m_dist2Site;
	FltArrTyp m_distDiff;
	
};

}

}

#endif
