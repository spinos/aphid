/*
 *  GenericTetraGrid.h
 * 
 *  tetrahedon grid of value Tv 
 *
 *  Created by zhang on 17-2-5.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APHID_TTG_GENERIC_TETR_GRID_H
#define APHID_TTG_GENERIC_TETR_GRID_H

#include <geom/ConvexShape.h>
#include <sdb/Array.h>
#include <vector>

namespace aphid {

namespace ttg {

template <typename Tv>
class GenericTetraGrid 
{
	Tv * m_value;
	Vector3F * m_pos;
    sdb::Coord4 * m_cells;
    int m_numPoints;
    int m_numCells;

public:
	GenericTetraGrid();
	virtual ~GenericTetraGrid();
    
    void create(int np, int nc);
	
	const int & numPoints() const;
    const int & numCells() const;
	const Vector3F & pos(const int & i) const;
	const Tv & value(const int & i) const;
    void getCell(cvx::Tetrahedron & tet,
                const int & i) const;
    const sdb::Coord4 & cellVertices(const int & i) const;
    
    void setPos(const Vector3F & v, const int & i);
	void setValue(const Tv & v, const int & i);
	void setCell(const sdb::Coord4 & v, const int & i);
	void setCell(const int & v0, 
                    const int & v1, 
                    const int & v2, 
                    const int & v3, const int & i);
	
protected:

private:
    void internalClear();
    
};

template <typename Tv>
GenericTetraGrid<Tv>::GenericTetraGrid()
{
    m_value = 0;
    m_pos = 0;
    m_cells = 0;
}

template <typename Tv>
GenericTetraGrid<Tv>::~GenericTetraGrid()
{
    internalClear();
}

template <typename Tv>
void GenericTetraGrid<Tv>::internalClear()
{
	if(m_value) delete[] m_value;
	if(m_pos) delete[] m_pos;
    if(m_cells) delete[] m_cells;
}

template <typename Tv>
void GenericTetraGrid<Tv>::create(int np, int nc)
{
    internalClear();
    m_numPoints = np;
    m_numCells = nc;
    m_pos = new Vector3F[np];
    m_value = new Tv[np];
    m_cells = new sdb::Coord4[nc];
}

template <typename Tv>
const int & GenericTetraGrid<Tv>::numPoints() const
{ return m_numPoints; }

template <typename Tv>
const Vector3F & GenericTetraGrid<Tv>::pos(const int & i) const
{ return m_pos[i]; }

template <typename Tv>
const Tv & GenericTetraGrid<Tv>::value(const int & i) const
{ return m_value[i]; }

template <typename Tv>
const int & GenericTetraGrid<Tv>::numCells() const
{ return m_numCells; }

template <typename Tv>
const sdb::Coord4 & GenericTetraGrid<Tv>::cellVertices(const int & i) const
{ return m_cells[i]; }

template <typename Tv>
void GenericTetraGrid<Tv>::getCell(cvx::Tetrahedron & tet,
                const int & i) const
{
    const sdb::Coord4 & c = cellVertices(i);
    tet.set(m_pos[c.x], 
            m_pos[c.y], 
            m_pos[c.z], 
            m_pos[c.w]);
}

template <typename Tv>
void GenericTetraGrid<Tv>::setPos(const Vector3F & v, const int & i)
{ m_pos[i] = v; }

template <typename Tv>
void GenericTetraGrid<Tv>::setValue(const Tv & v, const int & i)
{ m_value[i] = v; }

template <typename Tv>
void GenericTetraGrid<Tv>::setCell(const sdb::Coord4 & v, const int & i)
{ m_cells[i] = v; }

template <typename Tv>
void GenericTetraGrid<Tv>::setCell(const int & v0, 
                    const int & v1, 
                    const int & v2, 
                    const int & v3, const int & i)
{ setCell(sdb::Coord4(v0, v1, v2, v3), i); }

}

}
#endif

