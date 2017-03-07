/*
 *  CollisionGrid.h
 *  
 *  adaptive grid stores objects by size, larger ones in lower level cells 
 *
 *  Created by jian zhang on 11/28/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef APH_SDB_COLLISION_GRID_H
#define APH_SDB_COLLISION_GRID_H

#include <sdb/AdaptiveGrid3.h>
#include <sdb/Array.h>
#include "boost/date_time/posix_time/posix_time.hpp"

namespace aphid {

namespace sdb {

/// T as object, Tshape as collision shape
template<typename T, typename Tshape>
class CollisionNode {

	T * m_obj;
	Tshape * m_shape;
	
public:
	CollisionNode(const T * object);
	~CollisionNode();
	
private:

};

template<typename T, typename Tshape>
CollisionNode<T, Tshape>::CollisionNode(const T * object)
{
	T * m_obj = object;
	Tshape * m_shape = new Tshape;
}

template<typename T, typename Tshape>
CollisionNode<T, Tshape>::~CollisionNode()
{
	delete m_shape;
}

class CollisionCell : public Array<int, CollisionNode >, public AdaptiveGridCell {

typedef Array<int, CollisionNode > TParent;
	
public:
	CollisionCell(Entity * parent = NULL);
	virtual ~CollisionCell();
	
	virtual void clear();
	
	void countNodesInCell(int & it);
	
private:

};	

class CollisionGrid : public AdaptiveGrid3<CollisionCell, CollisionNode, 10 > {

typedef AdaptiveGrid3<CollisionCell, CollisionNode, 10 > TParent;

public:
	CollisionGrid(Entity * parent = NULL);
	virtual ~CollisionGrid();
	
	template<typename Tf, int Ndiv>
	void insertNodeAtLevel(int level,
							Tf & fintersect)
	{
		std::cout<<"\n CollisionGrid::insertNodeAtLevel "<<level;
		std::cout.flush();
				
		boost::posix_time::ptime t0(boost::posix_time::second_clock::local_time());
		boost::posix_time::ptime t1;
		boost::posix_time::time_duration t3;
		
		GridSampler<Tf, CollisionNode, Ndiv > sampler;
		
		BoundingBox cellBx;
		
		int nc = 0;
		int c = 0;
		begin();
		while(!end() ) {
			if(key().w == level) {
			
				getCellBBox(cellBx, key() );
				sampler.sampleInBox(fintersect, cellBx, m_limitBox );
				
				const int & ns = sampler.numValidSamples();
				for(int i=0;i<ns;++i) {

					CollisionNode * par = new CollisionNode;
					
					const CollisionNode & src = sampler.sample(i);
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
	
	virtual void clear(); 
	
protected:
	
private:
	
};

}

}
#endif