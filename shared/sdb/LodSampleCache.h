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
		float radius;
		Vector3F nml;
		float noi;
	};
	
private:
	boost::scoped_array<ASample> m_data;
	int m_numSamples;
	
public:
	SampleCache();
	virtual ~SampleCache();
	
	void create(int n);
	void clear();
	
	const int & numSamples() const;
	
	const float * points() const;
	const float * normals() const;
	
	ASample * data();
	
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
	
protected:

private:
	
};

}

}
#endif