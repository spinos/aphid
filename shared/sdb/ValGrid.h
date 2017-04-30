/*
 *  ValGrid.h
 *  adaptive grid with value per cell
 *
 *  Created by jian zhang on 11/28/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef APH_SDB_VAL_GRID_H
#define APH_SDB_VAL_GRID_H

#include <sdb/AdaptiveGrid3.h>
#include <sdb/Array.h>

namespace aphid {

namespace sdb {

template<typename T>
class ValCell : public Array<int, T >, public AdaptiveGridCell {

typedef Array<int, T > TParent;
	
public:
	ValCell(Entity * parent = NULL);
	virtual ~ValCell();

private:
	
};

template<typename T>
ValCell<T>::ValCell(Entity * parent) : TParent(parent)
{}

template<typename T>
ValCell<T>::~ValCell()
{}
	
template<typename T>
class ValGrid : public AdaptiveGrid3<ValCell<T>, T, 10 > {

    typedef ValCell<T> TCell;
typedef AdaptiveGrid3<TCell, T, 10 > TParent;

public:
	ValGrid(Entity * parent = NULL);
	virtual ~ValGrid();
	
	void insertValueAtLevel(const int & level, 
	        const Vector3F & pref,
	        const T & v);
	
	void finishInsert();
	
protected:
	
private:

};

template<typename T>
ValGrid<T>::ValGrid(Entity * parent) : TParent(parent)
{}

template<typename T>
ValGrid<T>::~ValGrid()
{}

template<typename T>
void ValGrid<T>::insertValueAtLevel(const int & level, 
        const Vector3F & pref,
        const T & v)
{

    
    int i=0;
    while(i<=level) {
        const Coord4 c = cellCoordAtLevel(pref, i);
        TCell * cell = findCell(c);
        if(!cell) {
            cell = addCell(c);
        }
        
        if(i==level) {
#if 0
    std::cout<<"\n ValGrid::insertValueAtLevel "<<level<<" p "<<pref
        <<" v "<<v
        <<" cell "<<c;
#endif
            T * vr = new T;
            *vr = v;
            cell->insert(rand() & 127, vr);
        }
        i++;
    }
    
}

template<typename T>
void ValGrid<T>::finishInsert()
{
    storeCellNeighbors();
	storeCellChildren();
	calculateBBox();
}

}

}
#endif
