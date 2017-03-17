/*
 *  HeightBccGrid.h
 *  ttg
 *
 *  Created by jian zhang on 7/21/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TTG_HEIGHT_BCC_GRID_3_H
#define APH_TTG_HEIGHT_BCC_GRID_3_H

#include <ttg/AdaptiveBccGrid3.h>
#include <img/BoxSampleProfile.h>

namespace aphid {

namespace ttg {

class HeightSubdivCondition {

public:
	
	float m_sampleSize;
	float m_sigma;
	float m_height;
	Float2 m_deHeight;
	int m_numSamples;
	
	HeightSubdivCondition();
	
	void setSigma(const float & x);
	void setSampleSize(const float & x);
	bool satisfied(const float & py) const;
	
};

class HeightBccGrid : public AdaptiveBccGrid3 {

public:
	HeightBccGrid();
	virtual ~HeightBccGrid();
	
	template<typename Tf>
	void subdivide(const Tf & fld,
						sdb::AdaptiveGridDivideProfle & prof);
						
protected:
	
private:
/// randomly sample height in cell multiple times
	template<typename Tf>
	bool isCellDirty(const sdb::Coord4 & cellCoord,
				const Tf & fld, 
				HeightSubdivCondition & condition,
				img::BoxSampleProfile<float> * sampler) const;
	
};

template<typename Tf>
void HeightBccGrid::subdivide(const Tf & fld,
						sdb::AdaptiveGridDivideProfle & prof)
{
#if 0
	std::cout<<"\n HeightBccGrid::subdivide ";
#endif
	std::vector<sdb::Coord4 > divided;

	BoundingBox cb;
	float filterSize;
	
	img::BoxSampleProfile<float> sampler;
	sampler._channel = 0;
	sampler._defaultValue = 0.5f;

	HeightSubdivCondition condition;
	condition.setSigma(levelCellSize(prof.maxLevel() ) );
	bool stat;
	
	int level = prof.minLevel();
	while(level < prof.maxLevel() ) {
		condition.setSampleSize(levelCellSize(level) );
		filterSize = condition.m_sampleSize / fld.sampleSize();
		std::cout<<"\n level"<<level<<" filter size"<<filterSize;
		
		fld.getSampleProfile(&sampler, filterSize);
		std::cout<<"\n sample level "<<sampler._loLevel<<","<<sampler._hiLevel;
		
		std::vector<sdb::Coord4> dirty;
		begin();
		while(!end() ) {
			if(key().w == level) {
				
				if(level < 1) {
					stat = true;
				} else {
					stat = isCellDirty(key(),
							fld, condition, &sampler);					
				}
				
				if(stat) {
					dirty.push_back(key() );
				}
			}
			
			next();
		}
		
		if(dirty.size() < 1) {
			break;
		}
#if 0			
		std::cout<<"\n subdiv level "<<level<<" divided "<<divided.size();
#endif
		std::vector<sdb::Coord4>::const_iterator it = dirty.begin();
		for(;it!=dirty.end();++it) {
			getCellBBox(cb, *it);
			cb.expand(-1e-3f);
			BccCell3 * cell = findCell(*it);
			subdivideCellToLevel(cell, *it, cb, level+1, &divided);
		}
		level++;
	}
	
	enforceBoundary(divided);
	storeCellNeighbors();
	storeCellChildren();
}

template<typename Tf>
bool HeightBccGrid::isCellDirty(const sdb::Coord4 & cellCoord,
				const Tf & fld, 
				HeightSubdivCondition & condition,
				img::BoxSampleProfile<float> * sampler) const
{
	const Vector3F cc = cellCenter(cellCoord );
	Vector3F q;
	for(int i=0;i<condition.m_numSamples;++i) {
		q = cc + Vector3F(RandomFn11() * condition.m_sampleSize, 
					RandomFn11() * condition.m_sampleSize, 
					RandomFn11() * condition.m_sampleSize);
					
		sampler->_uCoord = (q.x + 1024.f) / 2048.f; 
		sampler->_vCoord = (q.z + 1024.f) / 2048.f;

		condition.m_height = fld.sampleHeight(sampler);
		condition.m_deHeight = fld.sampleHeightDerivative(sampler);
						
		if( condition.satisfied(q.y) ) {
			return true;
		}
	}
	return false;
}

}
}
#endif
