/*
 *  LodGrid.h
 *  
 *  adaptive grid with aggregated property per cell
 *
 *  Created by jian zhang on 11/28/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef APH_SDB_LOD_GRID_H
#define APH_SDB_LOD_GRID_H

#include <sdb/AdaptiveGrid3.h>
#include <sdb/Array.h>
#include <sdb/GridSampler.h>
#include <math/Quantization.h>
#include "boost/date_time/posix_time/posix_time.hpp"

namespace aphid {

namespace sdb {

/// splat with aggregated position, normal, radius
class LodNode {

public:
	LodNode();
	~LodNode();
	
	Vector3F pos; 
	int index;
	Vector3F nml;
	int geom;
	Float2 texcoord;
	int comp;
	int colcom;
	
private:

};

class LodCell : public Array<int, LodNode >, public AdaptiveGridCell {

typedef Array<int, LodNode > TParent;
	
public:
	LodCell(Entity * parent = NULL);
	virtual ~LodCell();
	
	virtual void clear();
	
	void countNodesInCell(int & it);
	void dumpNodesInCell(LodNode * dst);
	
	template<typename T>
	void closestToPoint(T * result);
	
	template<typename Ts>
	void dumpSamplesInCell(Ts * dst);
	
	template<typename T, typename Tshape>
	void selectInCell(Sequence<int> & indices, 
					T & selFilter,
					const Tshape & selShape);
	
private:
	
};

template<typename T>
void LodCell::closestToPoint(T * result)
{
	begin();
	while(!end() ) {
		LodNode * nd = value();
		float d = nd->pos.distanceTo(result->_toPoint);
		if(d < result->_distance) {
			result->_distance = d;
			result->_hasResult = true;
			result->_hitPoint = nd->pos;
			result->_hitNormal = nd->nml;
		}
		
		if(result->closeEnough() ) {
			return;
		}
		
		next();
	}
}

template<typename Ts>
void LodCell::dumpSamplesInCell(Ts * dst)
{
	float r, g, b, alpha;
	begin();
	while(!end() ) {
		LodNode * a = value();
		const int & i = a->index;
		
		Ts & d = dst[i];
		d.pos = a->pos;
		d.nml = a->nml;
		d.u = a->texcoord.x;
		d.v = a->texcoord.y;
		d.geomcomp = (a->geom<<22) | a->comp;
		col32::decodeC(r, g, b, alpha, a->colcom );
		d.col.set(r, g, b);
		
		next();
	}
}

template<typename T, typename Tshape>
void LodCell::selectInCell(Sequence<int> & indices, 
							T & selFilter,
							const Tshape & selShape)
{
	begin();
	while(!end() ) {
		LodNode * a = value();
		if(selShape.intersectPoint(a->pos) ) {
			if(selFilter.isRemoving() ) {
				indices.remove(a->index);
			} else { 
/// append or replace
				indices.insert(a->index);
			}
		}
		
		next();
	}
}
	

class LodGrid : public AdaptiveGrid3<LodCell, LodNode, 10 > {

typedef AdaptiveGrid3<LodCell, LodNode, 10 > TParent;

public:
	LodGrid(Entity * parent = NULL);
	virtual ~LodGrid();
	
/// level reached
	template<typename Tf>
	int subdivideToLevel(Tf & fintersect,
						AdaptiveGridDivideProfle & prof)
	{
#if 0
		std::cout<<"\n LodGrid::subdivide ";
#endif
		BoundingBox cb;
		int level = prof.minLevel();
		while(level < prof.maxLevel() ) {
/// dirty cells at level
			std::vector<Coord4> dirty;
			begin();
			while(!end() ) {
				if(key().w == level) {
					getCellBBox(cb, key() );
					
					if(limitBox().intersect(cb) ) {
						if(fintersect.intersect(cb) ) {
							dirty.push_back(key() );
						}
					}
					
				} else if(key().w > level) {
					break;
				} 
				
				next();
			}
			
			if(dirty.size() < 1) {
				break;
			}
#if 0
			std::cout<<"\n level"<<level;
			if(prof.m_dividedCoord) {
				std::cout<<" divided "<<prof.m_dividedCoord->size();
			}
#endif
			std::vector<Coord4>::const_iterator it = dirty.begin();
			for(;it!=dirty.end();++it) {
				subdivideCell(fintersect, *it);
			}
			level++;
		}
		storeCellNeighbors();
		storeCellChildren();
		
#if 0		
		const int nlc = numCellsAtLevel(level );
		std::cout<<"\n to level "<<level<<" n cell "<<nlc;
		std::cout.flush();
#endif
		return level;
	}
	
	template<typename Tf, int Ndiv>
	void insertNodeAtLevel(int level,
							Tf & fintersect)
	{
		std::cout<<"\n LodGrid::insertNodeAtLevel "<<level;
		std::cout.flush();
				
		boost::posix_time::ptime t0(boost::posix_time::second_clock::local_time());
		boost::posix_time::ptime t1;
		boost::posix_time::time_duration t3;
		
		GridSampler<Tf, LodNode, Ndiv > sampler;
		
		BoundingBox cellBx;
		const BoundingBox & limBx = limitBox();
		
		int nc = 0;
		int c = 0;
		begin();
		while(!end() ) {
			if(key().w == level) {
			
				getCellBBox(cellBx, key() );
				sampler.sampleInBox(fintersect, cellBx, limBx );
				
				const int & ns = sampler.numValidSamples();
				
				for(int i=0;i<ns;++i) {

					LodNode * par = new LodNode;
					
					const LodNode & src = sampler.sample(i);
					*par = src;
					par->index = -1;
					
					value()->insert(i, par);
					
				}
				c += ns;
				
				nc++;
				
				if((nc & 1023) == 0) {
					t1 = boost::posix_time::ptime (boost::posix_time::second_clock::local_time());
					t3 = t1 - t0;
					std::cout<<"\n processed "<<nc<<" cells in "<<t3.seconds()<<" seconds";
					std::cout.flush();
				}
				
			}
			
			if(key().w > level) {
				break;
			}
			
			next();
		}
		
		t1 = boost::posix_time::ptime (boost::posix_time::second_clock::local_time());
		t3 = t1 - t0;
		
		std::cout<<"\n done. "
				<<"\n n samples "<<c
				<<"\n cost time "<<t3.seconds()<<" seconds";
		std::cout.flush();
	}
	
	void inserNodedByAggregation(int minLevel, int maxLevel);
	void aggregateAtLevel(int level);
	
	int countLevelNodes(int level);
	void dumpLevelNodes(LodNode * dst, int level);
	
	virtual void clear();
	
protected:
	template<typename Ts>
	void dumpLevelSamples(Ts * dst, int level);
	
private:
	void aggregateInCell(LodCell * cell, 
						const Coord4 & cellCoord);
	void processKmean(int & n, 
					LodNode * samples,
					const float & separateDist);
					
	template<typename Tf>
	void subdivideCell(Tf & fintersect,
						const Coord4 & cellCoord)
	{
		LodCell * cell = findCell(cellCoord);
		if(!cell) {
			std::cout<<"\n [ERROR] LodGrid cannot find cell to subdivide "<<cellCoord;
			return;
		}
		
		if(cell->hasChild() ) {
			return;
		}
		
		BoundingBox cb;
		for(int i=0; i< 8; ++i) { 
			
			getCellChildBox(cb, i, cellCoord );
			if(limitBox().intersect(cb) ) {
				if(fintersect.intersect(cb) ) {
					subdivide(cellCoord, i);
				}
			}
			
		}
		
	}
	
};

template<typename Ts>
void LodGrid::dumpLevelSamples(Ts * dst, int level)
{
	begin();
	while(!end() ) {
		if(key().w == level) {
			value()->dumpSamplesInCell(dst);
		}
		
		if(key().w > level) {
			break;
		}

		next();
	}
}

}

}
#endif