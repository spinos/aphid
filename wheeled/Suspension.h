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
	class Profile {
	public:
		Profile();		
		float _upperWishboneAngle[2];
		float _lowerWishboneAngle[2];
		float _wheelHubX;
		float _wheelHubR;
		float _upperJointY, _lowerJointY;
		float _upperWishboneLength, _lowerWishboneLength;
		float _upperWishboneTilt, _lowerWishboneTilt;
		bool _steerable, _powered;
	};
	
	Suspension();
	virtual ~Suspension();
	void setProfile(const Profile & info);
	const float width() const;
	const float wheelHubX() const;
	const bool isPowered() const;
	
	btRigidBody* create(const Vector3F & pos, bool isLeft = true);
	void connectWheel(btRigidBody* hub, btRigidBody* wheel, bool isLeft);
	void powerDrive(const float & speed, const float & wheelR);
	
	static float RodRadius;
	static btRigidBody * ChassisBody;
	static Vector3F ChassisOrigin;
private:
	btRigidBody* createCarrier(const Matrix44F & tm, bool isLeft);
	btRigidBody* Suspension::createWishbone(const Matrix44F & tm, bool isUpper, bool isLeft);
	btCompoundShape* createWishboneShape(bool isUpper, bool isLeft);
	const Matrix44F wishboneHingTMLocal(bool isUpper, bool isLeft, bool isFront) const;
	void wishboneLA(bool isUpper, bool isLeft, bool isFront, float & l, float & a) const;
	void connectArm(btRigidBody* arm, const Matrix44F & tm, bool isUpper, bool isLeft, bool isFront);
	void applyBrake(bool enable);
	void applyMotor(float rps);
	Profile m_profile;
	btGeneric6DofConstraint* m_steerJoint[2];
	btGeneric6DofConstraint* m_driveJoint[2];
};
}