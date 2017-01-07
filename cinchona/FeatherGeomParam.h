/*
 *  FeatherGeomParam.h
 *  cinchona
 *
 *  Created by jian zhang on 1/3/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef FEATHER_GEOM_PARAM_H
#define FEATHER_GEOM_PARAM_H

class Geom1LineParam;

class FeatherGeomParam {
	
/// 0  flying
/// 1:2 upper covert
/// 3:4 lower covert 
	Geom1LineParam * m_lines[5];
	
public:
	FeatherGeomParam();
	~FeatherGeomParam();
	
	void setFlying(const int * nps,
						const float * chords,
						const float * ts);
	void setCovert(int i,
						const int * nps,
						const float * chords,
						const float * ts);
						
	bool isChanged() const;
	const float & longestChord() const;
	Geom1LineParam * line(int i);
	
private:
	
};
	
#endif