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
#include <PlantCommon.h>
#include "ExampVox.h"

namespace aphid {

class Plant;

class ForestCell : public sdb::Array<sdb::Coord2, Plant> {

	ForestCell * m_cellNeighbor[26];
	
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
	
	template<typename T, typename Tshape>
	void selectSamples(T & selFilter, const Tshape & shape);
	
	template<typename T>
	void processFilter(T & selFilter);
	
	void deselectSamples();
	const int & numActiveSamples() const;
	const int * activeSampleIndices() const;
	const int & numVisibleSamples() const;
	const int * visibleSampleIndices() const;
	
	void clearSamples();
	void clearPlants();
	
	bool hasSamples(int level) const;
	
	const sdb::SampleCache * sampleCacheAtLevel(int level) const;
	
	template<typename Tf>
	void assignSamplePlantType(const Tf & selFilter);
	
	template<typename Tf>
	void colorSampleByPlantType(const Tf & selFilter);
/// 0-25
	void setCellNeighbor(ForestCell * v, int idx);
	
	ForestCell * cellNeighbor(int idx);
	
	template<typename Tgrd, typename Tctx>
	bool collide(Tgrd * grid, Tctx * context);
	
	template<typename T>
	int countInstances(T* tforest);
	
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
	sdb::AdaptiveGridDivideProfle subdprof;
	subdprof.setLevels(0, maxLevel);
	const int reached = m_lodsamp->subdivideToLevel<T>(ground, subdprof);
	if(reached < maxLevel) {
		return;
	}
	m_lodsamp->insertNodeAtLevel<Tc, 4 >(maxLevel, closestGround);
	m_lodsamp->buildSampleCache(maxLevel, maxLevel);	
	const int & nv = m_lodsamp->numSamplesAtLevel(maxLevel);
	if(nv < 1) {
		m_activeSampleIndices.reset();
		m_visibleSampleIndices.reset();
		return;
	} 
	
	m_activeSampleIndices.reset(new int[nv]);
	m_visibleSampleIndices.reset(new int[nv]);
	
	assignSamplePlantType<Tf>(selFilter);
	
}

template<typename T, typename Tshape>
void ForestCell::selectSamples(T & selFilter, const Tshape & shape)
{
	if(selFilter.isReplacing() ) {
		deselectSamples();
	}
	
	m_lodsamp->select(m_activeInd, selFilter, shape);
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

template<typename Tf>
void ForestCell::assignSamplePlantType(const Tf & selFilter)
{
	sdb::SampleCache * cache = m_lodsamp->samplesAtLevel(selFilter.maxSampleLevel() );
	sdb::SampleCache::ASample * sps = cache->data();
	const int & ns = cache->numSamples();
	for(int i=0; i<ns; ++i) {
		sdb::SampleCache::ASample & asp = sps[i];
		const int k = rand() & 65535;
		asp.exmInd = selFilter.selectPlantType(k);
		asp.col = selFilter.plantTypeColor(asp.exmInd);
	}
}

template<typename Tf>
void ForestCell::colorSampleByPlantType(const Tf & selFilter)
{
	sdb::SampleCache * cache = m_lodsamp->samplesAtLevel(selFilter.maxSampleLevel() );
	sdb::SampleCache::ASample * sps = cache->data();
	const int & ns = cache->numSamples();
	for(int i=0; i<ns; ++i) {
		sdb::SampleCache::ASample & asp = sps[i];
		asp.col = selFilter.plantTypeColor(asp.exmInd);
	}
}

template<typename T, typename Tctx>
bool ForestCell::collide(T * forest, Tctx * context)
{
	if(isEmpty() ) {
		return false;
	}
	
	bool doCollide;
	
	begin();
	while(!end()) {
		PlantData * d = value()->index;
		if(d == NULL) {
			throw "ForestCell has null data";
		}
		
		if(context->_minIndex > -1) {
			doCollide = key().x < context->_minIndex;
		} else {
			doCollide = true;
		}
		
		if(doCollide) {
			float scale = d->t1->getSide().length() * .5f;
			if(context->contact(d->t1->getTranslation(),
							forest->plantSize(key().y) * scale) ) {
				return true;
			}
		}
		
		next();
	}
	
	return false;
}
	
template<typename T>
int ForestCell::countInstances(T* tforest)
{
	int cnt = 0;
	bool bCompound = false;
	int iExample = -1;
	ExampVox * v = 0;
	begin();
	while(!end() ) {
		const int & curK = key().y;
		if(iExample != curK) {
			iExample = curK;
			v = tforest->plantExample(iExample );
			if(v) {
				bCompound = v->isCompound();
			} else {
				std::cout<<"\n exmp is null "<<iExample;
			}
		}
		
		if(v) {
			if(bCompound) {
				cnt += v->numInstances();
				
			} else {
				cnt++;
			}
		}
		
		next();
	}
	return cnt;
}

}
#endif
