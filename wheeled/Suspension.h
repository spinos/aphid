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
			_upperWishboneAngle[1] = .79f;
			_lowerWishboneAngle[0] = -.36f;
			_lowerWishboneAngle[1] = .79f;
			_wheelHubX = .6f;
			_wheelHubR = 1.41f;
			_upperJointY = 1.73f; 
			_lowerJointY = -1.f;
			_upperWishboneLength = 2.7f;
			_lowerWishboneLength = 5.3f;
			_upperWishboneTilt = -.2f;
			_lowerWishboneTilt = .1f;
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
	const float wheelHubX() const;
	
	btRigidBody* create(const Vector3F & pos, bool isLeft = true);
	
	static float RodRadius;
	static btRigidBody * ChassisBody;
	static Vector3F ChassisOrigin;
private:
	btRigidBody* createCarrier(const Matrix44F & tm, bool isLeft);
	btRigidBody* createUpperWishbone(const Vector3F & pos, bool isLeft);
	btRigidBody* createLowerWishbone(const Vector3F & pos, bool isLeft);
	btCompoundShape* createWishboneShape(bool isUpper, bool isLeft);
	const Matrix44F wishboneHingTMLocal(bool isUpper, bool isLeft, bool isFront) const;
	void wishboneLA(bool isUpper, bool isLeft, bool isFront, float & l, float & a) const;
	void connectArm(btRigidBody* arm, const Vector3F & pos, bool isUpper, bool isLeft, bool isFront);
	Profile m_profile;
};
}