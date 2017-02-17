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
		int geomInd;
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
	
	void setColorByNoise();
	
	static const int DataStride;
	
};

class LodSampleCache : public LodGrid {

/// level 0 - 7
	SampleCache m_samples[8];
	
public:
	LodSampleCache(Entity * parent = NULL);
	virtual ~LodSampleCache();
	
	bool hasLevel(int x);
	void buildSamples(int minLevel, int maxLevel);
	
	const int & numSamplesAtLevel(int x) const;
	
	const SampleCache * samplesAtLevel(int x) const;
	SampleCache * samplesAtLevel(int x);
	
protected:

private:
	
};

}

}
#endif