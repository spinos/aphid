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

	void getMeanValue(T & dst);
	void getFirstValue(T & dst);
	
private:
	
};

template<typename T>
ValCell<T>::ValCell(Entity * parent) : TParent(parent)
{}

template<typename T>
ValCell<T>::~ValCell()
{}

template<typename T>
void ValCell<T>::getMeanValue(T & dst)
{
	dst.setZero();
	int cnt = 0;
	TParent::begin();
	while(!TParent::end() ) {
		dst += *TParent::value();
		cnt++;
		TParent::next();
	}
	
	if(cnt > 0) {
		dst /= (float)cnt;
	}
}

template<typename T>
void ValCell<T>::getFirstValue(T & dst)
{
	TParent::begin();
	dst = *TParent::value();
}
	
template<typename T>
class ValGrid : public AdaptiveGrid3<ValCell<T>, T, 10 > {

public:
    typedef ValCell<T> TCell;
private:
typedef AdaptiveGrid3<TCell, T, 10 > TParent;

public:
	ValGrid(Entity * parent = NULL);
	virtual ~ValGrid();
	
	void insertValueAtLevel(const int & level, 
	        const Vector3F & pref,
	        const T & v);
	
	void finishInsert();
	
	void getCellColor(float * dst);
	void getFirstValue(T & dst);
	void getMeanValue(T & dst);
	
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
        const Coord4 c = TParent::cellCoordAtLevel(pref, i);
        TCell * cell = TParent::findCell(c);
        if(!cell) {
            cell = TParent::addCell(c);
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
    TParent::storeCellNeighbors();
	TParent::storeCellChildren();
	TParent::calculateBBox();
}

template<typename T>
void ValGrid<T>::getCellColor(float * dst)
{
	T mv;
	TParent::value()->getMeanValue(mv);
	mv.getColor(dst);
}

template<typename T>
void ValGrid<T>::getFirstValue(T & dst)
{
	TParent::value()->getFirstValue(dst);
}

template<typename T>
void ValGrid<T>::getMeanValue(T & dst)
{
	TParent::value()->getMeanValue(dst);
}

}

}
#endif
