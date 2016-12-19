/*
 *  PCASimilarity.h
 *  
 *  similarity search via pca
 *
 *  Created by jian zhang on 12/20/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_PCA_SIMILARITY_H
#define APH_PCA_SIMILARITY_H

#include <math/linearMath.h>

namespace aphid {

template<typename T, typename Tf>
class PCASimilarity {

	std::vector<Tf * > m_features;
	
public:
	PCASimilarity();
	virtual ~PCASimilarity();
	
/// set example by pnts, initialize search
	void begin(const std::vector<Vector3F > & pnts);
/// select a feature
	bool select(const std::vector<Vector3F > & pnts);
	
	void computeSimilarity();
	
/// n row of first feature
	int featureDim() const;
	
	int numFeatures() const;
	
protected:

private:
	void clearFeatures();
	
};

template<typename T, typename Tf>
PCASimilarity<T, Tf>::PCASimilarity()
{}

template<typename T, typename Tf>
PCASimilarity<T, Tf>::~PCASimilarity()
{
	clearFeatures();
}

template<typename T, typename Tf>
void PCASimilarity<T, Tf>::begin(const std::vector<Vector3F > & pnts)
{
	clearFeatures();
	
	const int np = pnts.size();
	Tf * af = new Tf(np);
	for(int i=0;i<np;++i) {
		af->setPnt((const T *)&pnts[i], i);
	}
	m_features.push_back(af);
}

template<typename T, typename Tf>
bool PCASimilarity<T, Tf>::select(const std::vector<Vector3F > & pnts)
{
	const int exnp = featureDim();
	if(exnp<1) {
		return false;
	}
	const int np = pnts.size();
	if(np < exnp/2 || np > (exnp + exnp/2) ) {
		return false;
	}
	
	Tf * af = new Tf(exnp);
	
	int unp = np;
	if(np < exnp) {
/// lack of points
		af->copy(*m_features[0]);
	} else if(np > exnp) {
/// more than enough points
		unp = exnp;
	}
	
	for(int i=0;i<unp;++i) {
		af->setPnt((const T *)&pnts[i], i);
	}
	m_features.push_back(af);
}

template<typename T, typename Tf>
void PCASimilarity<T, Tf>::clearFeatures()
{
	const int n = numFeatures();
	for(int i=0;i<n;++i) {
		delete m_features[i];
	}
	m_features.clear();
}

template<typename T, typename Tf>
int PCASimilarity<T, Tf>::featureDim() const
{
	if(numFeatures() < 1) {
		return 0;
	}
	
	return m_features[0]->numPnts();
}

template<typename T, typename Tf>
int PCASimilarity<T, Tf>::numFeatures() const
{ return m_features.size(); }

template<typename T, typename Tf>
void PCASimilarity<T, Tf>::computeSimilarity()
{
	const int n = numFeatures();
	for(int i=0;i<n;++i) {
		m_features[i]->toLocalSpace();
	}
}

}

#endif
