/*
 *  ForestCell.h
 *  proxyPaint
 *
 *  plant and sample storage in cell
 *  selected samples are active
 *  filtered are visible
 *
 *  Created by jian zhang on 1/19/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_FOREST_CELL_H
#define APH_FOREST_CELL_H

#include <sdb/Array.h>
#include <sdb/LodSampleCache.h>

namespace aphid {

class Plant;

class ForestCell : public sdb::Array<sdb::Coord2, Plant> {

	sdb::Sequence<int> m_activeInd;
	boost::scoped_array<int> m_activeSampleIndices;
	boost::scoped_array<int> m_visibleSampleIndices;
	sdb::LodSampleCache * m_lodsamp;
	int m_numActiveSamples;
	int m_numVisibleSamples;
	
public:
	ForestCell(Entity * parent = NULL);
	virtual ~ForestCell();
	
	template<typename T, typename Tc, typename Tf>
	void buildSamples(T & ground, Tc & closestGround,
					Tf & selFilter,
					const BoundingBox & cellBox);

	void reshuffleSamples(const int & level);
	
	const float * samplePoints(int level) const;
	const float * sampleNormals(int level) const;
	const float * sampleColors(int level) const;
	const int & numSamples(int level) const;
	
	template<typename T>
	void selectSamples(T & selFilter);
	
	template<typename T>
	void processFilter(T & selFilter);
	
	void deselectSamples();
	const int & numActiveSamples() const;
	const int * activeSampleIndices() const;
	const int & numVisibleSamples() const;
	const int * visibleSampleIndices() const;
	
	void clearSamples();
	
	bool hasSamples(int level) const;
	
	const sdb::SampleCache * sampleCacheAtLevel(int level) const;
	
protected:
	 
private:
	void updateIndices(int & count, int * indices,
			sdb::Sequence<int> & srcInd);
	
};

template<typename T, typename Tc, typename Tf>
void ForestCell::buildSamples(T & ground, Tc & closestGround,
					Tf & selFilter,
					const BoundingBox & cellBox)
{
	//std::cout<<"\n ForestCell::buildSamples"<<cellBox;
	const int & maxLevel = selFilter.maxSampleLevel();
	m_lodsamp->fillBox(cellBox, selFilter.sampleGridSize() );
	const int reached = m_lodsamp->subdivideToLevel<T>(ground, 0, maxLevel);
	if(reached < maxLevel) {
		return;
	}
	m_lodsamp->insertNodeAtLevel<Tc, 4 >(maxLevel, closestGround);
	m_lodsamp->buildSampleCache(maxLevel, maxLevel);
	const int & nv = m_lodsamp->numSamplesAtLevel(maxLevel);
	if(nv) {
		m_activeSampleIndices.reset(new int[nv]);
		m_visibleSampleIndices.reset(new int[nv]);
	}
}

template<typename T>
void ForestCell::selectSamples(T & selFilter)
{
	if(selFilter.isReplacing() ) {
		deselectSamples();
	}
	
	m_lodsamp->select(m_activeInd, selFilter);
	updateIndices(m_numActiveSamples,
				m_activeSampleIndices.get(),
				m_activeInd);
}

template<typename T>
void ForestCell::processFilter(T & selFilter)
{	
	const sdb::SampleCache * cache = sampleCacheAtLevel(selFilter.maxSampleLevel() );
	const sdb::SampleCache::ASample * sps = cache->data();
	const int * activeInd = activeSampleIndices();
	
	m_numVisibleSamples = 0;
	for(int i=0;i<m_numActiveSamples;++i) {
		const int & ind = activeInd[i];
		const sdb::SampleCache::ASample & asp = sps[ind];
		const float & k = asp.noi;

		bool stat = selFilter.throughPortion(k);
		
		if(stat) {
			stat = selFilter.throughImage(asp.noi, asp.u, asp.v);
		}
		
		if(stat) {
			stat = selFilter.throughNoise3D(asp.pos);
		}
		
		if(stat) {
			m_visibleSampleIndices[m_numVisibleSamples] = ind;
			m_numVisibleSamples++;
		}
		
	}
	
}

}
#endif
