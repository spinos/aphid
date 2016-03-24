/*
 *  MinMaxBin.h
 *  kdtree
 *
 *  Created by jian zhang on 10/27/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

namespace aphid {

#define MMBINNSPLITLIMIT 33
#define MMBINNSPLITLIMITP1 34
#define MMBINNSPLITLIMITM1 32
#define MMBINNSPLITLIMITM1F 31.9999f

class MinMaxBins {
/// n-1+2 splits first and last is bound
///       each has left and right count
/// n     bins
///
///    0  1  2       n-1  n
///    |  |  |       |    |    pos
///    0  1  2       n-1  n    split
///  0  1  2  3  n-1    n      min, left to split
///     0  1  2       n-1  n   max, right to split

	float m_pos[MMBINNSPLITLIMITP1];
    unsigned m_minBin[MMBINNSPLITLIMIT];
	unsigned m_maxBin[MMBINNSPLITLIMIT];
	int m_numSplits;
	float m_delta;
	bool m_isEven;
	bool m_isFlat;

public:
	MinMaxBins();
	~MinMaxBins();
	
	void reset(const float & lft, const float & rgt);
	void createEven(const float & lft, const float & rgt);
	void add(const float &min, const float &max);
	void scan();
	void getCounts(const unsigned &idx, unsigned &left, unsigned &right) const;
	bool isFlat() const;
	void setFlat();
	const float & delta() const;
	bool insertSplitPos(const float & x);
	void printSplitPos() const;
	const float & splitPos(const int & idx) const;
	float leftEmptyDistance(int & idx, const int & head = 0) const;
	float rightEmptyDistance(int & idx, const int & tail = 0) const;
	
	int firstSplitToRight(const float & x) const;
	int lastSplitToLeft(const float & x) const;
	
	const int & numSplits() const;
	bool isEmpty() const;
	bool isFull() const;
	int maxNumSplits() const;
	void verbose() const;
	
private:
	const float & firstSplitPos() const;
	const float & lastSplitPos() const;
	void getGrid(const float & x, int & lft, int & rgt) const;
};

}
