/*
 *  MinMaxBins.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/27/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "MinMaxBins.h"
#include <iostream>

namespace aphid {
    
bool MinMaxBins::UnqunatizedPosition = true;

MinMaxBins::MinMaxBins() : 

m_numSplits(0),
m_isEven(false),
m_isFlat(0)
{}

MinMaxBins::~MinMaxBins() 
{}

/// split0         split1
///        bin0                 
/// 0              1   n
/// |              |bound
void MinMaxBins::reset(const float & lft, const float & rgt)
{ 
	m_numSplits = 2;
	m_delta = (rgt - lft) / MMBINNSPLITLIMITM1F;
	m_pos[0] = lft;
	m_pos[1] = rgt; 
// keep the right bound
	m_pos[MMBINNSPLITLIMIT] = rgt;
	int i = 0;
	for(; i < MMBINNSPLITLIMIT; ++i) {
		m_minBin[i] = m_maxBin[i] = 0;
	}
}

bool MinMaxBins::insertSplitPos(const float & x)
{ 
	if(isFull() ) return false;

/// out of bound
	if(x < m_pos[0] + m_delta) return false;
	if(x > m_pos[MMBINNSPLITLIMIT] - m_delta) return false;

#if 0	
	int i, j;
	for(i=1;i<m_numSplits;++i) {
/// move forward
/// no need for integer coord
    if(UnqunatizedPosition) {
		if(x < m_pos[i] + m_delta && i < m_numSplits - 1) {
			if(x>m_pos[i-1] + m_delta)
				m_pos[i] = x;
				return true;
		}
	}		
		if(x > (m_pos[i-1] + m_delta) 
			&& x < (m_pos[i] - m_delta) ) {
/// push following
			for(j=m_numSplits; j>i;--j ) {
				m_pos[j] = m_pos[j-1];
			}
/// insert at
			m_pos[i] = x;
			m_numSplits++;
			m_pos[m_numSplits-1] = m_pos[MMBINNSPLITLIMIT];
			return true;
		}
	}
#else
	int lft, rgt;
	getGrid(x, lft, rgt);
	if(UnqunatizedPosition) {
/// move right forward		
		if(rgt < m_numSplits-1) {
			if(x > m_pos[rgt] - m_delta 
				&& x>m_pos[lft] + m_delta) {
				m_pos[rgt] = x;
				return true;
			}
		}
	}
	
	if(x > (m_pos[lft] + m_delta) 
			&& x < (m_pos[rgt] - m_delta) ) {
/// push following
		for(int j=m_numSplits; j>rgt;--j ) {
			m_pos[j] = m_pos[j-1];
		}
/// insert at
		m_pos[rgt] = x;
		m_numSplits++;
		m_pos[m_numSplits-1] = m_pos[MMBINNSPLITLIMIT];
		return true;
	}
#endif
	return false;
}
	
void MinMaxBins::createEven(const float & lft, const float & rgt)
{
	m_numSplits = MMBINNSPLITLIMIT;
	m_delta = (rgt - lft) / MMBINNSPLITLIMITM1F;
	
	int i = 0;
	for(; i < MMBINNSPLITLIMIT; ++i) {
		m_minBin[i] = m_maxBin[i] = 0;
		m_pos[i] = lft + m_delta * i;
	}
	m_isEven = true;
}

///     <----->
///   i-1  i    i+1
///   |    |    |
void MinMaxBins::add(const float &lft, const float &rgt)
{
	m_minBin[firstSplitToRight(lft)]+=1;
	m_maxBin[lastSplitToLeft(rgt)]+=1;
}

void MinMaxBins::scan()
{
	int i;
	for(i = 1; i < m_numSplits; i++)
		m_minBin[i] += m_minBin[i - 1];
	for(i = m_numSplits - 2; i >= 0; i--)
		m_maxBin[i] += m_maxBin[i + 1];
}

void MinMaxBins::getCounts(const unsigned &idx, unsigned &left, unsigned &right) const
{
	left = m_minBin[idx];
	right =	m_maxBin[idx];
}

char MinMaxBins::isFlat() const
{ return m_isFlat; }

void MinMaxBins::setFlat()
{ m_isFlat = 1; }

const float & MinMaxBins::delta() const
{ return m_delta; }

const int & MinMaxBins::numSplits() const
{ return m_numSplits; }

bool MinMaxBins::isEmpty() const
{ return m_numSplits < 3; }

bool MinMaxBins::isFull() const
{ return m_numSplits == MMBINNSPLITLIMIT; }

int MinMaxBins::maxNumSplits() const
{ return MMBINNSPLITLIMIT; }

const float & MinMaxBins::splitPos(const int & idx) const
{ return m_pos[idx]; }

const float & MinMaxBins::firstSplitPos() const
{ return m_pos[0]; }

const float & MinMaxBins::lastSplitPos() const
{ return m_pos[m_numSplits-1]; }

float MinMaxBins::leftEmptyDistance(int & idx, const int & head) const
{
	if(m_minBin[0] > 1) return -1.f;
	if(m_minBin[1]==0) {
		idx = 1;
		return m_pos[1] - m_pos[0];
	}
	return -1.f;
}

float MinMaxBins::rightEmptyDistance(int & idx, const int & tail) const
{
	if(m_maxBin[m_numSplits-1] > 1) return -1.f;
	if(m_maxBin[m_numSplits-2]==0) {
		idx = m_numSplits-2;
		return m_pos[m_numSplits-1] - m_pos[m_numSplits-2];
	}
	return -1.f;
}

///         x <----
///     i-1     i    split
///     |       |     
int MinMaxBins::firstSplitToRight(const float & x) const
{
	if(x< firstSplitPos() ) return 0;
	if(x>= lastSplitPos() ) return m_numSplits-1;
	
#if 0
	//if(m_isEven) return (x - m_pos[0]) / m_delta + 1;
	int i;
	for(i=1;i<m_numSplits;++i) {
		if(x >= m_pos[i-1] && x < m_pos[i]) {
			return i;
		}
	}
	return m_numSplits-1;
#else
	int lft, rgt;
	getGrid(x, lft, rgt);
	return rgt;
#endif
}

///   ----> x 
///     i       i+1    split
///     |       |     
int MinMaxBins::lastSplitToLeft(const float & x) const
{
	if(x< firstSplitPos() ) return 0;
	if(x>= lastSplitPos() ) return m_numSplits-1;

#if 0	
	//if(m_isEven) return (x - m_pos[0]) / m_delta + 1;
	int i;
	for(i=0;i<m_numSplits-1;++i) {
		if(x >= m_pos[i] && x < m_pos[i+1]) {
			return i;
		}
	}
	return m_numSplits-1;
#else
	int lft, rgt;
	getGrid(x, lft, rgt);
	return lft;
#endif
}

void MinMaxBins::getGrid(const float & x, int & lft, int & rgt) const
{
	lft = 0;
	rgt = m_numSplits - 1;
	int mid = (lft + rgt)/2;
	float fm;
	while(rgt > lft+1) {
		fm = m_pos[mid];
		if(fm==x) {
			lft = mid;
			rgt = mid+1;
			return;
		}
		if(fm>x) rgt = mid;
		if(fm<x) lft = mid;
		mid = (lft + rgt)/2;
	}
}

void MinMaxBins::verbose() const
{
	int i=0;
	std::cout<<"\n min "<<m_numSplits<<" (";
	for(;i<m_numSplits;++i) {
		std::cout<<" "<<m_minBin[i];
	}
	std::cout<<")\n max "<<m_numSplits<<" (";
	for(i=0;i<m_numSplits;++i) {
		std::cout<<" "<<m_maxBin[i];
	}
	std::cout<<")\n";
	printSplitPos();
}

void MinMaxBins::printSplitPos() const
{
	std::cout<<"\n minmax bin delta "<<m_delta
	<<" split pos "<<m_numSplits<<" ( "<<m_pos[0];
	int i=1;
	for(;i<m_numSplits;++i) {
		std::cout<<", "<<m_pos[i];
	}
	std::cout<<" ) margin "<<m_pos[1] - m_pos[0]<<"/"<<m_pos[m_numSplits-1] - m_pos[m_numSplits-2];
}

}
