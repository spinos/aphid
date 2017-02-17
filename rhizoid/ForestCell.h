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
	
	template<typename T, typename Tc>
	void buildSamples(T & ground, Tc & closestGround,
					const BoundingBox & cellBox,
					const float & cellSize,
					int maxLevel = 4);
	
	const float * samplePoints(int level) const;
	const float * sampleNormals(int level) const;
	const int & numSamples(int level) const;
	
	template<typename T>
	void selectSamples(T & selFilter);
	
	void deselectSamples();
	const int & numSelectedSamples() const;
	
protected:
	const sdb::SampleCache * sampleAtLevel(int level) const;
	 
private:
};

template<typename T, typename Tc>
void ForestCell::buildSamples(T & ground, Tc & closestGround,
					const BoundingBox & cellBox,
					const float & cellSize,
					int maxLevel)
{
	if(m_lodsamp->hasLevel(maxLevel)) {
		return;
	}
	m_lodsamp->fillBox(cellBox, cellSize);
	m_lodsamp->subdivideToLevel<T>(ground, 0, maxLevel);
	m_lodsamp->insertNodeAtLevel<Tc, 4 >(maxLevel, closestGround);
	m_lodsamp->buildSamples(maxLevel, maxLevel);
	const int & nv = m_lodsamp->numSamplesAtLevel(maxLevel);
	if(nv) {
		m_activeSampleIndices.reset(new int[nv]);
	}
	deselectSamples();
}

template<typename T>
void ForestCell::selectSamples(T & selFilter)
{
	
}

}
#endif
