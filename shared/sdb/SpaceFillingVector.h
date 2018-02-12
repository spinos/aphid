/*
 *  SpaceFillingVector.h
 *  
 *  T is element type
 *  Tc is space filling curve type
 *
 *  Created by jian zhang on 2/12/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_SDB_SPACE_FILLING_VECTOR_H
#define APH_SDB_SPACE_FILLING_VECTOR_H

#include <vector>
#include <math/QuickSort.h>
#include <math/morton3.h>
#include <math/hilbert3.h>

namespace aphid {

namespace sdb {

template<typename T>
class SpaceFillingVector {

	std::vector<T> m_v;
	
public:
	SpaceFillingVector();
	virtual ~SpaceFillingVector();
	
	void push_back(const T& x);
	void clear();
	int size() const;
/// encode and sort
	template<typename Tc>
	void createSFC(const Tc& sfcf);
	
	void sort();
	
	bool searchSFC(int* dst, const int* range) const;
	
	int findElement(const int& k) const;
	
	T& operator[](int i);
	const T& operator[](int i) const;
	
protected:
	
	int searchElementNoGreaterThan(int first, int last, const int& thre) const;
	int searchElementNoLessThan(int first, int last, const int& thre) const;
	
private:

};

template<typename T>
SpaceFillingVector<T>::SpaceFillingVector()
{}

template<typename T>
SpaceFillingVector<T>::~SpaceFillingVector()
{ clear(); }

template<typename T>
void SpaceFillingVector<T>::push_back(const T& x)
{ m_v.push_back(x); }

template<typename T>
void SpaceFillingVector<T>::clear()
{ m_v.clear(); }

template<typename T>
int SpaceFillingVector<T>::size() const
{ return m_v.size(); }

template<typename T>
T& SpaceFillingVector<T>::operator[](int i)
{ return m_v[i]; }
	
template<typename T>
const T& SpaceFillingVector<T>::operator[](int i) const
{ return m_v[i]; }

template<typename T>
template<typename Tc>
void SpaceFillingVector<T>::createSFC(const Tc& sfcf)
{
	const int n = m_v.size();
	for(int i=0;i<n;++i) {
		T& vi = m_v[i];
		vi._key = sfcf.computeKey((const float*)&vi._pos);
	}
	sort();
}	

template<typename T>
void SpaceFillingVector<T>::sort()
{
	const int n = m_v.size();
	QuickSort1::SortVector<int, T>(m_v, 0, n-1);
}

template<typename T>
bool SpaceFillingVector<T>::searchSFC(int* dst, const int* range) const
{
	const int n = m_v.size();
	
	if(m_v[0]._key >= range[1] || m_v[n-1]._key < range[0]) {
		dst[0] = 0;
		dst[1] = -1;
		return false;
	}
	
	dst[0] = searchElementNoGreaterThan(0, n, range[0]);
	dst[1] = searchElementNoLessThan(0, n, range[1]);
	return (dst[1] > dst[0] + 1);
}

template<typename T>
int SpaceFillingVector<T>::searchElementNoGreaterThan(int first, int last, const int& thre) const
{
	int mid;
	while(last > first + 1) {
		
		mid = (first + last) / 2;
		
		const int& r = m_v[mid]._key;
		if(r == thre)
			return mid;
		
		if(r > thre)
			last = mid;
		else
			first = mid;
		
	}
	
	return first;
}

template<typename T>
int SpaceFillingVector<T>::searchElementNoLessThan(int first, int last, const int& thre) const
{
	int mid;
	while(last > first + 1) {
		
		mid = (first + last) / 2;
		
		const int& r = m_v[mid]._key;
		if(r == thre)
			return mid;
		
		if(r > thre)
			last = mid;
		else
			first = mid;
		
	}
	
	return last;
}

template<typename T>
int SpaceFillingVector<T>::findElement(const int& k) const
{
	if(m_v[0]._key == k)
		return 0;
		
	const int n = m_v.size();
	if(m_v[n-1]._key == k)
		return n-1;
		
	int i = searchElementNoLessThan(0, n-1, k);
	if(m_v[i]._key == k)
		return i;
		
	if(m_v[i-1]._key == k)
		return i-1;
		
	return -1;
}

struct FZFCurve {
	
	float _origin[3];
	float _h;
	float _oneoverh;
	int _range[2];
	
	void setRange(int xlow, int ylow, int zlow,
				int xhigh, int yhigh, int zhigh)
	{
		_range[0] = encodeMorton3D(xlow, ylow, zlow);
		_range[1] = encodeMorton3D(xhigh, yhigh, zhigh);
	}
	
	void setOrginSpan(const float& originx,
					const float& originy,
					const float& originz,
					const float& span) 
	{
		_origin[0] = originx;
		_origin[1] = originy;
		_origin[2] = originz;
		_h = span / 1024.f;
		_oneoverh = 1024.f / span;
	}
	
	int computeKey(const float* p) const
	{
		int x = (p[0] - _origin[0]) * _oneoverh;
		int y = (p[1] - _origin[1]) * _oneoverh;
		int z = (p[2] - _origin[2]) * _oneoverh;
		return encodeMorton3D(x, y, z);
	}
};

struct FHilbertRule {
	
	float _u0[3];
	float _red[3];
	float _green[3];
	float _blue[3];
	int _level;
	int _range[2];
	
	FHilbertRule() : _level(10) 
	{}
	
	void setRange(const float* p, const float* q, 
					const int& level)
	{
		_range[0] = hilbert3DCoord(p[0], p[1], p[2],
						_u0[0], _u0[1], _u0[2],
						_red[0], _red[1], _red[2],
						_green[0], _green[1], _green[2],
						_blue[0], _blue[1], _blue[2],
						level);
		_range[1] = hilbert3DCoord(q[0], q[1], q[2],
						_u0[0], _u0[1], _u0[2],
						_red[0], _red[1], _red[2],
						_green[0], _green[1], _green[2],
						_blue[0], _blue[1], _blue[2],
						level);
	}
	
	void setOriginSpan(const float& originx,
					const float& originy,
					const float& originz,
					const float& span) 
	{
		const float spanH = span * .5f;
		_u0[0] = originx + spanH;
		_u0[1] = originy + spanH;
		_u0[2] = originz + spanH;
		_red[0] = spanH;
		_red[1] = 0;
		_red[2] = 0;
		_green[0] = 0;
		_green[1] = spanH;
		_green[2] = 0;
		_blue[0] = 0;
		_blue[1] = 0;
		_blue[2] = spanH;
	}
	
	int computeKey(const float* p) const
	{
		return hilbert3DCoord(p[0], p[1], p[2],
						_u0[0], _u0[1], _u0[2],
						_red[0], _red[1], _red[2],
						_green[0], _green[1], _green[2],
						_blue[0], _blue[1], _blue[2],
						_level);
	}
	
	int computeKey(const float* p, const int& level) const
	{
		return hilbert3DCoord(p[0], p[1], p[2],
						_u0[0], _u0[1], _u0[2],
						_red[0], _red[1], _red[2],
						_green[0], _green[1], _green[2],
						_blue[0], _blue[1], _blue[2],
						level);
	}
	
	void getBox(float* centerHalfSpan) const
	{
		centerHalfSpan[0] = _u0[0];
		centerHalfSpan[1] = _u0[1];
		centerHalfSpan[2] = _u0[2];
		centerHalfSpan[3] = _red[0];
	}
	
	void computeChildCoord(float* childCoord, const int& i,
				const float* parentCoord) const
	{
		float& h = childCoord[3];
		h = parentCoord[3] / 2.f;
		childCoord[0] = parentCoord[0] + h * HilbertSubNodeCoord[i][0];
		childCoord[1] = parentCoord[1] + h * HilbertSubNodeCoord[i][1];
		childCoord[2] = parentCoord[2] + h * HilbertSubNodeCoord[i][2];
	}
	
};

}

}

#endif