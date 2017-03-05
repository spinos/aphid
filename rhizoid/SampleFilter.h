/*
 *  SampleFilter.h
 *  
 *
 *  Created by jian zhang on 2/18/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_SAMPLE_FILTER_H
#define APH_SAMPLE_FILTER_H

#include <math/Vector3F.h>
#include <geom/SelectionContext.h>
#include <math/ANoise3.h>
#include <boost/scoped_array.hpp>
#include <vector>
#include <map>

namespace aphid {
    
class ExrImage;

class SampleFilter : public ANoise3Sampler {

	std::map<int, bool> m_plantTypeMap;
	int m_maxSampleLevel;
	float m_sampleGridSize;
	float m_portion;
	SelectionContext::SelectMode m_mode;
	boost::scoped_array<int> m_plantTypeIndices;
	boost::scoped_array<Vector3F> m_plantTypeColors;
	int m_numPlantTypeIndices;
	int m_numPlantTypeColors;
	
public:
	SampleFilter();
	virtual ~SampleFilter();
	
	void setMode(SelectionContext::SelectMode mode);
	void setPortion(const float & x);
	
	bool isReplacing() const;
	bool isRemoving() const;
	bool isAppending() const;
	
	const int & maxSampleLevel() const;
	const float & sampleGridSize() const;
	void computeGridLevelSize(const float & cellSize,
				const float & sampleDistance);
				
	bool throughPortion(const float & x) const;
	bool throughNoise3D(const Vector3F & p) const;
	bool throughImage(const float & k, const float & s, const float & t) const;
	
	const float & portion() const;
	
	const ExrImage * m_imageSampler;
	
	void resetPlantTypeIndices(const std::vector<int> & indices);
	void resetPlantTypeColors(const std::vector<Vector3F> & colors);

	const Vector3F & plantTypeColor(int idx) const;
	
/// randomly
	int selectPlantType(int x) const;
	
	std::map<int, bool> * plantTypeMap();
	
protected:

private:
	void initPlantTypeIndices();
	void initPlantTypeColors();
	
};

}
#endif