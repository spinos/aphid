/*
 *  Suspension.h
 *  wheeled
 *
 *  Created by jian zhang on 5/31/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
namespace caterpillar {
class Suspension {
public:
	struct Profile {
		Profile() {
			_upperWishboneAngle[0] = -.26f;
			_upperWishboneAngle[1] = .26f;
			_lowerWishboneAngle[0] = -.26f;
			_lowerWishboneAngle[1] = .26f;
			_wheelHubX = 1.f;
			_upperJointY = 1.73f; 
			_lowerJointY = -1.f;
			_upperWishboneLength = 4.f;
			_lowerWishboneLength = 4.f;
			_upperWishboneTilt = 0.f;
			_lowerWishboneTilt = 0.f;
			_steerable = false;
		}
		
		float _upperWishboneAngle[2];
		float _lowerWishboneAngle[2];
		float _wheelHubX;
		float _upperJointY, _lowerJointY;
		float _upperWishboneLength, _lowerWishboneLength;
		float _upperWishboneTilt, _lowerWishboneTilt;
		bool _steerable;
	};
	
	Suspension();
	virtual ~Suspension();
	void setProfile(const Profile & info);
	const float width() const;
	
	void create(const Vector3F & pos, bool isLeft = true);
private:
	Profile m_profile;
};
}