/*
 *  LodSampleCache.h
 *  
 *
 *  Created by jian zhang on 2/17/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_SDB_LOD_SAMPLE_CACHE_H
#define APH_SDB_LOD_SAMPLE_CACHE_H

#include <sdb/LodGrid.h>
#include <boost/scoped_array.hpp>

namespace aphid {

namespace sdb {

class SampleCache {

public:
	struct ASample {
		Vector3F pos;
		float noi;
		Vector3F nml;
		float u;
		Vector3F col;
		float v;
		int geomcomp;
		int exmInd;
		int pad0;
		int pad1;
	};
	
private:
	boost::scoped_array<ASample> m_data;
	int m_numSamples;
	
public:
	SampleCache();
	virtual ~SampleCache();
	
	void create(int n);
	void clear();
	void assignNoise();
	
	const int & numSamples() const;
	
	const float * points() const;
	const float * normals() const;
	const float * colors() const;
	
	ASample * data();
	const ASample * data() const;
	const ASample & getASample(const int & i) const;
	
	void setColorByNoise();
	void setColorByUV();
	
	static const int DataStride;
	
};

class LodSampleCache : public LodGrid {

/// level 0 - 7
	SampleCache m_samples[8];
	
public:
	LodSampleCache(Entity * parent = NULL);
	virtual ~LodSampleCache();
	
	bool hasLevel(int x);
	void buildSampleCache(int minLevel, int maxLevel);
	
	const int & numSamplesAtLevel(int x) const;
	
	const SampleCache * samplesAtLevel(int x) const;
	SampleCache * samplesAtLevel(int x);
	
	template<typename T>
	void select(Sequence<int> & indices,
				T & selFilter);
				
	void reshuffleAtLevel(const int & level);
				
	virtual void clear();
	
protected:

private:
	template<typename T>
	void selectChildCell(Sequence<int> & indices,
				const Coord4 & cellCoord,
				LodCell * cell,
				T & selFilter,
				const int & level,
				const int & maxLevel);
				
};

template<typename T>
void LodSampleCache::select(Sequence<int> & indices,
				T & selFilter)
{
	const int & maxLevel = selFilter.maxSampleLevel();
	BoundingBox cellBx;
	begin();
	while(!end() ) {
		
		const Coord4 & k = key();
		
		getCellBBox(cellBx, k );
		if(selFilter.intersect(cellBx) ) {
			selectChildCell(indices, k, value(), selFilter, 0, maxLevel);
		}
		
		if(k.w > 0) {
			break;
		}
		next();
	}
	
}

template<typename T>
void LodSampleCache::selectChildCell(Sequence<int> & indices,
				const Coord4 & cellCoord,
				LodCell * cell,
				T & selFilter,
				const int & level,
				const int & maxLevel)
{
	if(level == maxLevel) {
		cell->selectInCell(indices, selFilter);
		return;
	}
	
	if(!cell->hasChild() ) {
		return;
	}
	
	BoundingBox cb;
	Coord4 cc;
	for(int i=0;i<8;++i) {
		AdaptiveGridCell * childCellD = cell->child(i);
		if(!childCellD ) {
			continue;
		}
		
		LodCell * childCell = static_cast<LodCell *>(childCellD);
		
		cc = childCoord(cellCoord, i);
		getCellBBox(cb, cc);
		if(selFilter.intersect(cb) ) {
			selectChildCell(indices, cc, childCell, selFilter, level + 1, maxLevel);
		}
	}
}

}

}
#endif