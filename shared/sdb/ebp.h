/*
 *  ebp.h
 *  
 *  energy based particle
 *
 *  Created by jian zhang on 11/28/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_SDB_EBP_H
#define APH_SDB_EBP_H

#include <sdb/AdaptiveGrid3.h>
#include <sdb/Array.h>
#include <boost/scoped_array.hpp>

namespace aphid {

class EbpNode {

public:
	EbpNode();
	~EbpNode();
	
	Vector3F pos;
	int index;
	
private:

};

class EbpCell : public sdb::Array<int, EbpNode >, public sdb::AdaptiveGridCell {

typedef sdb::Array<int, EbpNode > TParent;
	
public:
	EbpCell(Entity * parent = NULL);
	virtual ~EbpCell();
	
	virtual void clear();
	
private:

};

class EbpGrid : public sdb::AdaptiveGrid3<EbpCell, EbpNode, 10 > {

typedef sdb::AdaptiveGrid3<EbpCell, EbpNode, 10 > TParent;

	boost::scoped_array<Vector3F > m_pos;
	float m_repelDistance;
	
public:
	EbpGrid();
	virtual ~EbpGrid();
	
/// reset level0 cell size and bound
	void fillBox(const BoundingBox & b,
				const float & h);
		
	template<typename Tf>
	void subdivideToLevel(Tf & fintersect,
						int minLevel, int maxLevel)
	{
		BoundingBox cb;
		int level = minLevel;
		while(level < maxLevel) {
			std::vector<sdb::Coord4> dirty;
			begin();
			while(!end() ) {
				if(key().w == level) {
					getCellBBox(cb, key() );
					
					if(fintersect.intersect(cb) )
						dirty.push_back(key() );
				}
				next();
			}
			
			// std::cout<<"\n level"<<level<<" divd "<<dirty.size();
			
			std::vector<sdb::Coord4>::const_iterator it = dirty.begin();
			for(;it!=dirty.end();++it) {
				subdivideCell(fintersect, *it);
			}
			level++;
		}
		m_repelDistance = .67f / levelCellSize(level);
		storeCellNeighbors();
	}
	
	void insertNodeAtLevel(int level);
	void cachePositions();
	const Vector3F * positions() const;
	
	const int & numParticles() const;
	
	void update();
	virtual void clear(); 
			
private:
	template<typename Tf>
	void subdivideCell(Tf & fintersect,
						const sdb::Coord4 & cellCoord)
	{
		EbpCell * cell = findCell(cellCoord);
		if(!cell) {
			std::cout<<"\n [ERROR] EbpGrid cannot find cell to subdivide "<<cellCoord;
			return;
		}
		
		if(cell->hasChild() ) 
			return;
			
		BoundingBox cb;
		int i;	
	/// add 8 children
		for(i=0; i< 8; ++i) { 
			getCellChildBox(cb, i, cellCoord );
			
			if(fintersect.intersect(cb) )
				subdivide(cell, cellCoord, i);
		}
	}
	
	void extractPos(Vector3F * dst);
	void repelForce(Vector3F & frepel,
					EbpCell * cell,
					const EbpNode * node);
	void repelForceInCell(Vector3F & frepel,
					EbpCell * cell,
					const EbpNode * node);
	
};

}
#endif