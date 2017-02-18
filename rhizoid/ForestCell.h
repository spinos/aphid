/*
 *  ForestCell.h
 *  proxyPaint
 *
 *  plant and sample storage in cell
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

	sdb::Sequence<int> m_activeSampleKeys;
	boost::scoped_array<int> m_activeSampleIndices;
	sdb::LodSampleCache * m_lodsamp;
	int m_numActiveSamples;
	
public:
	ForestCell(Entity * parent = NULL);
	virtual ~ForestCell();
	
	template<typename T, typename Tc, typename Tf>
	void buildSamples(T & ground, Tc & closestGround,
					Tf & selFilter,
					const BoundingBox & cellBox);
	template<typename T, typename Tc, typename Tf>
	void rebuildSamples(T & ground, Tc & closestGround,
					Tf & selFilter,
					const BoundingBox & cellBox);
	
	const float * samplePoints(int level) const;
	const float * sampleNormals(int level) const;
	const int & numSamples(int level) const;
	
	template<typename T>
	bool selectSamples(T & selFilter);
	
	void deselectSamples();
	const int & numSelectedSamples() const;
	const int * selectedSampleIndices() const;
	
	void clearSamples();
	
	bool hasSamples(int level) const;
	
protected:
	const sdb::SampleCache * sampleAtLevel(int level) const;
	 
private:
	void updateActiveIndices();
	
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
	m_lodsamp->buildSamples(maxLevel, maxLevel);
	const int & nv = m_lodsamp->numSamplesAtLevel(maxLevel);
	if(nv) {
		m_activeSampleIndices.reset(new int[nv]);
	}
}

template<typename T, typename Tc, typename Tf>
void ForestCell::rebuildSamples(T & ground, Tc & closestGround,
					Tf & selFilter,
					const BoundingBox & cellBox)
{
	deselectSamples();
	clearSamples();
	buildSamples(ground, closestGround, selFilter,
					cellBox);
}

template<typename T>
bool ForestCell::selectSamples(T & selFilter)
{
	if(selFilter.isReplacing() ) {
		deselectSamples();
	}
	
	m_lodsamp->select(m_activeSampleKeys, selFilter);
	updateActiveIndices();
	return m_numActiveSamples > 0;
}

}
#endif
