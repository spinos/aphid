/*
 *  Suspension.h
 *  wheeled
 *
 *  Created by jian zhang on 5/31/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Common.h>
namespace caterpillar {
class Suspension {
public:
	struct Profile {
		Profile() {
			_upperWishboneAngle[0] = -.36f;
			_upperWishboneAngle[1] = .39f;
			_lowerWishboneAngle[0] = -.36f;
			_lowerWishboneAngle[1] = .39f;
			_wheelHubX = 1.f;
			_wheelHubR = 1.41f;
			_upperJointY = 1.73f; 
			_lowerJointY = -1.f;
			_upperWishboneLength = 3.1f;
			_lowerWishboneLength = 4.3f;
			_upperWishboneTilt = -0.1f;
			_lowerWishboneTilt = 0.07f;
			_steerable = false;
		}
		
		float _upperWishboneAngle[2];
		float _lowerWishboneAngle[2];
		float _wheelHubX;
		float _wheelHubR;
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
	
	static float RodRadius;
private:
	void createCarrier(const Matrix44F & tm, bool isLeft);
	void createUpperWishbone(const Matrix44F & tm, bool isLeft);
	void createLowerWishbone(const Matrix44F & tm, bool isLeft);
	btCompoundShape* createWishboneShape(bool isUpper, bool isLeft);
	Profile m_profile;
};
}