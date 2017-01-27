/*
 *  TetrahedronGrid.h
 *  
 *  tetrahedon grid of value Tv and order N
 *  Created by jian zhang on 10/28/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APHID_TETRAHEDRON_GRID_H
#define APHID_TETRAHEDRON_GRID_H

#include <ConvexShape.h>
namespace aphid {

template <int N>
class TetrahedronGridUtil 
{
	
public:
	TetrahedronGridUtil();
	
	static void CalculatePos(Vector3F * pos,
				const cvx::Tetrahedron & teta);
	
	static const int Ng;
	static Float4 * Wei;
	
private:
	void initWei();
	
};

template <int N>
TetrahedronGridUtil<N>::TetrahedronGridUtil()
{
	initWei();
}

template <int N>
Float4 * TetrahedronGridUtil<N>::Wei = new Float4[( ( N + 1 ) * ( N + 2 ) * ( N + 3 ) ) / 6];

template <int N>
const int TetrahedronGridUtil<N>::Ng = ( ( N + 1 ) * ( N + 2 ) * ( N + 3 ) ) / 6;

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
void TetrahedronGridUtil<N>::CalculatePos(Vector3F * pos,
				const cvx::Tetrahedron & teta)
{
	int i;
  int ii;
  int j;
  int k;
  int l;
  int p;

  p = 0;

  for ( i = 0; i <= N; i++ ) {
    for ( j = 0; j <= N - i; j++ ) {
      for ( k = 0; k <= N - i - j; k++ ) {
        l = N - i - j - k;

		const Float4 & wei = TetrahedronGridUtil<N>::Wei[p];
		
		pos[p] = teta.X(0) * wei.x
						+ teta.X(1) * wei.y
						+ teta.X(2) * wei.z
						+ teta.X(3) * wei.w;
						
        p = p + 1;
      }
    }
  }

}

template <typename Tv, int N>
class TetrahedronGrid 
{
	Tv * m_value;
	Vector3F * m_pos;
	
public:
	TetrahedronGrid(const cvx::Tetrahedron & tetra);
	virtual ~TetrahedronGrid();
	
	int numPoints() const;
	const Vector3F & pos(const int & i) const;
	const Tv & value(const int & i) const;
	
protected:

private:

};

template <typename Tv, int N>
TetrahedronGrid<Tv, N>::TetrahedronGrid(const cvx::Tetrahedron & tetra)
{
	int ng = numPoints();
	m_value = new Tv[ng];
	m_pos = new Vector3F[ng];
	
	TetrahedronGridUtil<N>::CalculatePos(m_pos, tetra);
	
}

template <typename Tv, int N>
TetrahedronGrid<Tv, N>::~TetrahedronGrid()
{
	delete[] m_value;
	delete[] m_pos;
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

}
#endif
