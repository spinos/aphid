/*
 *  LegendreVoxel.h
 *  
 *  legendre polynomial fitting in a voxel
 *  sample range by space filling curve
 *
 *  Created by jian zhang on 2/13/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TTG_LEGENDRE_SVO_RULE_H
#define APH_TTG_LEGENDRE_SVO_RULE_H

#include <math/LegendreInterpolation.h>
#include <sdb/SpaceFillingVector.h>
#include <ttg/UniformDensity.h>

namespace aphid {

namespace ttg {

template<typename Tc>
class LegendreSVORule {

/// space filling curve
	Tc m_sfc;
/// to build
	int m_maxLevel;
	
public:

	LegendreSVORule();
	
	void setMaxLevel(int x);
	
	Tc& sfc();
	const Tc& sfc() const;
	
/// T is svo node
/// Ts is sample
/// coord as (center, half_span)
	template<typename T, typename Ts>
	void buildRoot(T& node, float* coord,
			const sdb::SpaceFillingVector<Ts>& samples);
			
	template<typename T, typename Ts>
	void buildInner(T& node,
			const sdb::SpaceFillingVector<Ts>& samples);
	
	void computeChildCoord(float* coord, const float* parentCoord, const int& i);
	int computeFinestKey(const float* coord);
	
/// 8 subnode key and coord
/// sort by key
	template<typename T>
	void computeChildKeys(T* child, T& parent);
	
/// spawn i-th child from parent
/// return false if no sample found in range
	template<typename T, typename Ts>
	bool spawnNode(T* child, const int& i, T& parent,
			const sdb::SpaceFillingVector<Ts>& samples);
		
	template<typename T>
	bool endSubdivide(const T& node) const;
	
	template<typename T>
	static void PrintNode(const T& node);
	
	static void PrintCoord(const float* c);
	
/// builer Tb to traverser Tt
	template<typename Tt, typename Tb>
	static void SaveRootNode(Tt& t, float* rootCoord,
					const Tb& b);
					
	template<typename Tt, typename Tb>
	static void SaveNode(Tt& t, const Tb& b);
	
protected:

private:
	
};

template<typename Tc>
LegendreSVORule<Tc>::LegendreSVORule() :
m_maxLevel(4)
{}

template<typename Tc>
void LegendreSVORule<Tc>::setMaxLevel(int x)
{ m_maxLevel = x; }

template<typename Tc>
Tc& LegendreSVORule<Tc>::sfc()
{ return m_sfc; }

template<typename Tc>
const Tc& LegendreSVORule<Tc>::sfc() const
{ return m_sfc; }

template<typename Tc>
template<typename T, typename Ts>
void LegendreSVORule<Tc>::buildRoot(T& node, float* coord,
			const sdb::SpaceFillingVector<Ts>& samples)
{
	m_sfc.getBox(node._coord);
/// be the first one
	node._key = -1;
	node._level = 0;
	node._nextKey = (1<<31)-1;
	memset(node._ind, 0, 40);
	node._range[0] = 0;
	node._range[1] = samples.size();
	m_sfc.getBox(coord);
}

template<typename Tc>
template<typename T, typename Ts>
void LegendreSVORule<Tc>::buildInner(T& node,
			const sdb::SpaceFillingVector<Ts>& samples)
{
}

template<typename Tc>
template<typename T>
bool LegendreSVORule<Tc>::endSubdivide(const T& node) const
{
	bool stat = node._level >= m_maxLevel;
	return stat;
}

template<typename Tc>
void LegendreSVORule<Tc>::computeChildCoord(float* coord, const float* parentCoord, const int& i)
{ m_sfc.computeChildCoord(coord, i, parentCoord); }

template<typename Tc>
int LegendreSVORule<Tc>::computeFinestKey(const float* coord)
{ return m_sfc.computeKey(coord, 10); }

template<typename Tc>
template<typename T>
void LegendreSVORule<Tc>::computeChildKeys(T* child, T& parent) 
{
	const int level1 = parent._level + 1;
	for(int i=0;i<8;++i) {
		T& ci = child[i];
		computeChildCoord(ci._coord, parent._coord, i);
		ci._key = m_sfc.computeKey(ci._coord, level1);
		ci._level = level1;
	}
	
	QuickSort1::SortByKey<int, T>(child, 0, 7);
	
	for(int i=0;i<7;++i) {
		child[i]._nextKey = child[i+1]._key;
	}
	child[7]._nextKey = parent._nextKey;
	
}

template<typename Tc>
template<typename T, typename Ts>
bool LegendreSVORule<Tc>::spawnNode(T* child, const int& i, T& parent,
			const sdb::SpaceFillingVector<Ts>& samples)
{
	int ks[2];
	ks[0] = child[i]._key;
	
	if(i<7) {
		ks[1] = child[i+1]._key;
	} else {
		ks[1] = parent._nextKey;
	}
	
	bool stat = samples.searchSFC(child[i]._range, ks);
	
	return stat;
}

template<typename Tc>
template<typename T>
void LegendreSVORule<Tc>::PrintNode(const T& node)
{
	std::cout<<"  k "<<node._key
		<<"  l "<<node._level
		<<"\n    p "<<node.parentInd()
		<<" c ("<<node._ind[1]
		<<","<<node._ind[2]
		<<","<<node._ind[3]
		<<","<<node._ind[4]
		<<","<<node._ind[5]
		<<","<<node._ind[6]
		<<","<<node._ind[7]
		<<","<<node._ind[8]
		<<") d "<<node._ind[9]
		<<"  r ("<<node._range[0]
		<<","<<node._range[1]
		<<")";
}

template<typename Tc>
void LegendreSVORule<Tc>::PrintCoord(const float* c)
{ 
	std::cout<<" coord ("<<c[0]
		<<","<<c[1]
		<<","<<c[2]
		<<","<<c[3]
		<<") ";
}

template<typename Tc>
template<typename Tt, typename Tb>
void LegendreSVORule<Tc>::SaveRootNode(Tt& t, float* rootCoord,
						const Tb& b)
{
	memcpy(rootCoord, b._coord, 16);
	b.getChildInds(t._ind);
	t._ind[8] = -1;
	b.getLocation(t._ind[9]);
}

template<typename Tc>
template<typename Tt, typename Tb>
void LegendreSVORule<Tc>::SaveNode(Tt& t, const Tb& b)
{
	b.getChildInds(t._ind);
	b.getParent(t._ind[8]);
	b.getLocation(t._ind[9]);
}

}

}

#endif