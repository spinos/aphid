/*
 *  MinMaxBin.h
 *  kdtree
 *
 *  Created by jian zhang on 10/27/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

class MinMaxBins {
public:
	MinMaxBins();
	~MinMaxBins();
	
	void create(const unsigned &num, const float &min, const float &max);
	void add(const float &min, const float &max);
	void scan();
	void get(const unsigned &idx, unsigned &left, unsigned &right) const;
	char isFlat() const;
	void setFlat();
	const float & delta() const;
	
private:
	void validateIdx(int &idx) const;
	unsigned *m_minBin;
	unsigned *m_maxBin;
	unsigned m_binSize;
	float m_boundLeft, m_delta;
	char m_isFlat;
};
