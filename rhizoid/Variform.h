/*
 *  Variform.h
 *  
 *
 *  Created by jian zhang on 5/20/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef RHIZ_VARIFORM_H
#define RHIZ_VARIFORM_H

namespace aphid {

class Vector3F;

class Variform {

public:
	enum Pattern {
		pnRandom = 0,
		pnAngleAlign = 1
	};
	
	Variform();
	virtual ~Variform();
	
	void setPattern(Pattern x);
	void setShortPattern(short x);
	const Pattern & getPattern() const;
	
protected:
	static float deltaAnglePerGroup();
	static int NumAngleGroups;
	static int NumEventsPerGroup;
	
	int selectByAngle(const Vector3F & surfaceNml,
				const Vector3F & frameUp,
				Vector3F & modSide) const;
	void rotateSide(Vector3F & modSide,
				const Vector3F & frameSide,
				const float & alpha) const;
	
private:
	Pattern m_pattern;
	
};

}

#endif