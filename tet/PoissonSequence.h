/*
 *  PoissonSequence.h
 *  foo
 *
 *  Created by jian zhang on 6/19/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef TTG_POISSON_SEQUENCE_H
#define TTG_POISSON_SEQUENCE_H

#include <WorldGrid.h>
#include <Array.h>

namespace ttg {

template<typename T>
class PoissonSequence : public aphid::sdb::WorldGrid<aphid::sdb::Array<int, T >, T >
{

public:
	PoissonSequence();
	virtual ~PoissonSequence();
	
	bool reject(const T * cand);
	bool rejectIn(aphid::sdb::Array<int, T > * cell, const T * cand);
	
	void getCellBoundingBox(aphid::BoundingBox * dst);
	
protected:

private:
	
};

template<typename T>
PoissonSequence<T>::PoissonSequence() {}

template<typename T>
PoissonSequence<T>::~PoissonSequence() {}

template<typename T>
bool PoissonSequence<T>::reject(const T * cand)
{
	const aphid::sdb::Coord3 cellC = 
			aphid::sdb::WorldGrid<aphid::sdb::Array<int, T >, T >::gridCoord((const float *)&cand->pos);
	
	aphid::sdb::Array<int, T > * cell = 
			aphid::sdb::WorldGrid<aphid::sdb::Array<int, T >, T >::findCell(cellC);
			
	if(cell) {
		if(rejectIn(cell, cand) ) return true;
	}
	
	int i=0;
	for(;i<26;++i) {
		cell = aphid::sdb::WorldGrid<aphid::sdb::Array<int, T >, T >::findCell(
					aphid::sdb::Coord3(cellC.x + aphid::sdb::WorldGrid<aphid::sdb::Array<int, T >, T >::TwentySixNeighborCoord[i][0],
										cellC.y + aphid::sdb::WorldGrid<aphid::sdb::Array<int, T >, T >::TwentySixNeighborCoord[i][1],
										cellC.z + aphid::sdb::WorldGrid<aphid::sdb::Array<int, T >, T >::TwentySixNeighborCoord[i][2]) );
										
		if(cell) {
			if(rejectIn(cell, cand) ) return true;
		}
	}
	
	return false;
}

template<typename T>
bool PoissonSequence<T>::rejectIn(aphid::sdb::Array<int, T > * cell, const T * cand)
{
	cell->begin();
	while(!cell->end() ) {
		
		if(cand->collide(cell->value() ) )
			return true;
			
		cell->next();
	}
	return false;
}

template<typename T>
void PoissonSequence<T>::getCellBoundingBox(aphid::BoundingBox * dst)
{
	aphid::sdb::Array<int, T > * cell = aphid::sdb::WorldGrid<aphid::sdb::Array<int, T >, T >::value();
	dst->reset();
	cell->begin();
	while(!cell->end() ) {
		
		dst->expandBy(cell->value()->pos);
			
		cell->next();
	}
}

}
#endif
