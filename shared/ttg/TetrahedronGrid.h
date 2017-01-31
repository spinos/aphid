/*
 *  TetrahedronGrid.h
 *  
 *  tetrahedon grid of value Tv and order N
 *
 *  Created by jian zhang on 10/28/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APHID_TETRAHEDRON_GRID_H
#define APHID_TETRAHEDRON_GRID_H

#include <ConvexShape.h>
#include <sdb/Array.h>
#include <vector>

namespace aphid {

template <int N>
class TetrahedronGridUtil 
{
	
public:
	TetrahedronGridUtil();
	
	static void BuildVertices(sdb::Array<sdb::Coord3, int> & verts,
                Vector3F * pos,
				const cvx::Tetrahedron & teta,
                const int & firstVertex);
	static void BuildCells(std::vector<sdb::Coord4> & tets,
                sdb::Array<sdb::Coord3, int> & verts);

	static const int Ng;
	static Float4 * Wei;
	
private:
	void initWei();
    static bool findCellInd(sdb::Coord4 & dst,
                                const sdb::Coord3 v,
                                const int & i,
                                sdb::Array<sdb::Coord3, int> & verts);
    static sdb::Coord3 * sunitTetraOffset;
	
};

template <int N>
TetrahedronGridUtil<N>::TetrahedronGridUtil()
{
	initWei();
    sunitTetraOffset[0] = sdb::Coord3(1, 0, 0);
    sunitTetraOffset[1] = sdb::Coord3(0, 1, 0);
    sunitTetraOffset[2] = sdb::Coord3(0, 0, 1);
    sunitTetraOffset[3] = sdb::Coord3(0, 0, 0);
    
    sunitTetraOffset[4] = sdb::Coord3(0, 0, 1);
    sunitTetraOffset[5] = sdb::Coord3(0, 1, 0);
    sunitTetraOffset[6] = sdb::Coord3(1, 0, 0);
    sunitTetraOffset[7] = sdb::Coord3(1, 0, 1);
    
    sunitTetraOffset[8] = sdb::Coord3(0, 1, 0);
    sunitTetraOffset[9] = sdb::Coord3(1, 0, 1);
    sunitTetraOffset[10] = sdb::Coord3(0, 0, 1);
    sunitTetraOffset[11] = sdb::Coord3(0, 1, 1);
    
    sunitTetraOffset[12] = sdb::Coord3(0, 1, 0);
    sunitTetraOffset[13] = sdb::Coord3(1, 0, 0);
    sunitTetraOffset[14] = sdb::Coord3(1, 0, 1);
    sunitTetraOffset[15] = sdb::Coord3(1, 1, 0);
    
    sunitTetraOffset[16] = sdb::Coord3(1, 0, 1);
    sunitTetraOffset[17] = sdb::Coord3(1, 1, 0);
    sunitTetraOffset[18] = sdb::Coord3(0, 1, 1);
    sunitTetraOffset[19] = sdb::Coord3(0, 1, 0);
    
    sunitTetraOffset[20] = sdb::Coord3(0, 1, 1);
    sunitTetraOffset[21] = sdb::Coord3(1, 1, 0);
    sunitTetraOffset[22] = sdb::Coord3(1, 0, 1);
    sunitTetraOffset[23] = sdb::Coord3(1, 1, 1);
}

template <int N>
Float4 * TetrahedronGridUtil<N>::Wei = new Float4[( ( N + 1 ) * ( N + 2 ) * ( N + 3 ) ) / 6];

template <int N>
const int TetrahedronGridUtil<N>::Ng = ( ( N + 1 ) * ( N + 2 ) * ( N + 3 ) ) / 6;

template <int N>
sdb::Coord3 * TetrahedronGridUtil<N>::sunitTetraOffset = new sdb::Coord3[24];

template <int N>
void TetrahedronGridUtil<N>::initWei()
{
	int i;
  int ii;
  int j;
  int k;
  int l;
  int p = 0;

  for ( i = 0; i <= N; i++ ) {
    for ( j = 0; j <= N - i; j++ ) {
      for ( k = 0; k <= N - i - j; k++ ) {
        l = N - i - j - k;

		Float4 & wei = Wei[p];
		wei.x = (float)(i) / (float)(N);
		wei.y = (float)(j) / (float)(N);
		wei.z = (float)(k) / (float)(N);
		wei.w = (float)(l) / (float)(N);
		
        p = p + 1;
      }
    }
  }
}

template <int N>
void TetrahedronGridUtil<N>::BuildVertices(sdb::Array<sdb::Coord3, int> & verts,
                Vector3F * pos,
				const cvx::Tetrahedron & teta,
                const int & firstVertex)
{
	int i;
  int j;
  int k;
  int p;

  p = 0;

  for ( i = 0; i <= N; i++ ) {
    for ( j = 0; j <= N - i; j++ ) {
      for ( k = 0; k <= N - i - j; k++ ) {

		const Float4 & wei = TetrahedronGridUtil<N>::Wei[p];
		
		pos[p] = teta.X(0) * wei.x
						+ teta.X(1) * wei.y
						+ teta.X(2) * wei.z
						+ teta.X(3) * wei.w;
                        
        sdb::Coord3 t0(i, j, k);
            int * ind = new int;
            *ind = p + firstVertex;
            verts.insert(t0, ind);
						
        p++;
      }
    }
  }

}

template <int N>
bool TetrahedronGridUtil<N>::findCellInd(sdb::Coord4 & dst,
                                const sdb::Coord3 v,
                                const int & i,
                                sdb::Array<sdb::Coord3, int> & verts)
{
    int j = i<<2;
    sdb::Coord3 t0 = v + sunitTetraOffset[j];
    int * i0 = verts.find(t0);
    if(!i0) {
        return false;
    }
    
    sdb::Coord3 t1 = v + sunitTetraOffset[j+1];
    int * i1 = verts.find(t1);
    if(!i1) {
        return false;
    }
    
    sdb::Coord3 t2 = v + sunitTetraOffset[j+2];
    int * i2 = verts.find(t2);
    if(!i2) {
        return false;
    }
    
    sdb::Coord3 t3 = v + sunitTetraOffset[j+3];
    int * i3 = verts.find(t3);
    if(!i3) {
        return false;
    }
    
    dst = sdb::Coord4(*i0, *i1, *i2, *i3);
            
    return true;
}

template <int N>
void TetrahedronGridUtil<N>::BuildCells(std::vector<sdb::Coord4> & tets,
                            sdb::Array<sdb::Coord3, int> & verts)
{
    sdb::Coord4 tetind;
    	int i;
  int j;
  int k;
  
  for ( i = 0; i < N; i++ ) {
    for ( j = 0; j < N - i; j++ ) {
      for ( k = 0; k < N - i - j; k++ ) {
      
            const sdb::Coord3 ijk(i, j, k);
            for(int p=0;p<6;++p) {
                bool found = findCellInd(tetind, ijk, p, verts);
            
                if(found) {
                    tets.push_back(tetind);
                }
            }
            
          }
        }
    }

}

template <typename Tv, int N>
class TetrahedronGrid 
{
	Tv * m_value;
	Vector3F * m_pos;
    sdb::Coord4 * m_cells;
    int m_numCells;
    int m_vertexOffset;
	
public:
	TetrahedronGrid(const cvx::Tetrahedron & tetra,
                    const int & firstVertex = 0);
	virtual ~TetrahedronGrid();
	
	int numPoints() const;
    const int & numCells() const;
	const Vector3F & pos(const int & i) const;
	const Tv & value(const int & i) const;
    void getCell(cvx::Tetrahedron & tet,
                const int & i) const;
    const sdb::Coord4 & cellVertices(const int & i) const;
                
    void setValue(const Tv & v,
                const int & i);
                
    void setNodeDistance(const float & v, const int & i);
	
protected:

private:

};

template <typename Tv, int N>
TetrahedronGrid<Tv, N>::TetrahedronGrid(const cvx::Tetrahedron & tetra,
                                const int & firstVertex)
{
    m_vertexOffset = firstVertex;
    
	int ng = numPoints();
	m_value = new Tv[ng];
	m_pos = new Vector3F[ng];
	
    sdb::Array<sdb::Coord3, int> vind;
	TetrahedronGridUtil<N>::BuildVertices(vind, m_pos, tetra, firstVertex);
	
    std::vector<sdb::Coord4> cind;
    TetrahedronGridUtil<N>::BuildCells(cind, vind);
    
    m_numCells = cind.size();
    m_cells = new sdb::Coord4[m_numCells];
    for(int i=0;i<m_numCells;++i) {
        m_cells[i] = cind[i];
    }
    
    vind.clear();
}

template <typename Tv, int N>
TetrahedronGrid<Tv, N>::~TetrahedronGrid()
{
	delete[] m_value;
	delete[] m_pos;
    delete[] m_cells;
}

template <typename Tv, int N>
int TetrahedronGrid<Tv, N>::numPoints() const
{ return TetrahedronGridUtil<N>::Ng; }

template <typename Tv, int N>
const Vector3F & TetrahedronGrid<Tv, N>::pos(const int & i) const
{ return m_pos[i]; }

template <typename Tv, int N>
const Tv & TetrahedronGrid<Tv, N>::value(const int & i) const
{ return m_value[i]; }

template <typename Tv, int N>
const int & TetrahedronGrid<Tv, N>::numCells() const
{ return m_numCells; }

template <typename Tv, int N>
const sdb::Coord4 & TetrahedronGrid<Tv, N>::cellVertices(const int & i) const
{ return m_cells[i]; }

template <typename Tv, int N>
void TetrahedronGrid<Tv, N>::getCell(cvx::Tetrahedron & tet,
                const int & i) const
{
    const sdb::Coord4 & c = cellVertices(i);
    tet.set(m_pos[c.x - m_vertexOffset], 
            m_pos[c.y - m_vertexOffset], 
            m_pos[c.z - m_vertexOffset], 
            m_pos[c.w - m_vertexOffset]);
}

template <typename Tv, int N>
void TetrahedronGrid<Tv, N>::setValue(const Tv & v,
                const int & i)
{
    m_value[i] = v;
}

template <typename Tv, int N>
void TetrahedronGrid<Tv, N>::setNodeDistance(const float & v, const int & i)
{
    m_value[i]._distance = v;
}

}
#endif
