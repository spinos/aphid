/*
 *  GenericHexagonGrid.h
 *  
 *
 *  Created by jian zhang on 2/26/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APHID_GENERIC_HEXAGON_GRID_H
#define APHID_GENERIC_HEXAGON_GRID_H

#include <geom/ConvexShape.h>

namespace aphid {

template <typename Tv>
class GenericHexagonGrid {

	Tv * m_value;
	Vector3F * m_pos;
    int * m_cells;
    int m_numPoints;
    int m_numCells;
	
public:
	GenericHexagonGrid();
	virtual ~GenericHexagonGrid();
	
	void create(int np, int nc);
	
	const int & numPoints() const;
    const int & numCells() const;
	
	const Vector3F & pos(const int & i) const;
	const Tv & value(const int & i) const;
    const int * cellVertices(const int & i) const;
	void getCell(cvx::Hexagon & hexa, const int & i) const;
	
	void setPos(const Vector3F & v, const int & i);
	void setValue(const Tv & v, const int & i);
	void setCell(const int * v, const int & i);
				
protected:

private:
	void internalClear();
	
};

template <typename Tv>
GenericHexagonGrid<Tv>::GenericHexagonGrid()
{
    m_value = 0;
    m_pos = 0;
    m_cells = 0;
}

template <typename Tv>
GenericHexagonGrid<Tv>::~GenericHexagonGrid()
{
    internalClear();
}

template <typename Tv>
void GenericHexagonGrid<Tv>::internalClear()
{
	if(m_value) delete[] m_value;
	if(m_pos) delete[] m_pos;
    if(m_cells) delete[] m_cells;
}

template <typename Tv>
void GenericHexagonGrid<Tv>::create(int np, int nc)
{
    internalClear();
    m_numPoints = np;
    m_numCells = nc;
    m_pos = new Vector3F[np];
    m_value = new Tv[np];
    m_cells = new int[nc<<3];
}

template <typename Tv>
const int & GenericHexagonGrid<Tv>::numPoints() const
{ return m_numPoints; }

template <typename Tv>
const int & GenericHexagonGrid<Tv>::numCells() const
{ return m_numCells; }

template <typename Tv>
const Vector3F & GenericHexagonGrid<Tv>::pos(const int & i) const
{ return m_pos[i]; }

template <typename Tv>
const Tv & GenericHexagonGrid<Tv>::value(const int & i) const
{ return m_value[i]; }

template <typename Tv>
const int * GenericHexagonGrid<Tv>::cellVertices(const int & i) const
{ return &m_cells[i<<3]; }

template <typename Tv>
void GenericHexagonGrid<Tv>::getCell(cvx::Hexagon & hexa, const int & i) const
{
	const int * c = cellVertices(i);
	hexa.set(pos(c[0]), pos(c[1]), pos(c[2]), pos(c[3]),
			pos(c[4]), pos(c[5]), pos(c[6]), pos(c[7]));
}

template <typename Tv>
void GenericHexagonGrid<Tv>::setPos(const Vector3F & v, const int & i)
{ m_pos[i] = v; }

template <typename Tv>
void GenericHexagonGrid<Tv>::setValue(const Tv & v, const int & i)
{ m_value[i] = v; }

template <typename Tv>
void GenericHexagonGrid<Tv>::setCell(const int * v, const int & i)
{ memcpy(&m_cells[i<<3], v, 32); }

}
#endif
