/*
 *  Suspension.h
 *  wheeled
 *
 *  Created by jian zhang on 5/31/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "Damper.h"
#include "Wheel.h"
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
		float _steerArmJointZ;
		float _upperWishboneLength, _lowerWishboneLength;
		float _upperWishboneTilt, _lowerWishboneTilt;
		float _damperY;
		bool _steerable, _powered;
	};
	
	Suspension();
	virtual ~Suspension();
	void setProfile(const Profile & info);
	const float width() const;
	const float wheelHubX() const;
	const bool isPowered() const;
	const bool isSteerable() const;
	
	btRigidBody* create(const Vector3F & pos, bool isLeft = true);
	void connectWheel(Wheel * wheel, bool isLeft);
	
	void update();
	
	void computeDifferential(const Vector3F & turnAround, const float & z, const float & wheelSpan);
	void steer(const Vector3F & turnAround, const float & z, const float & wheelSpan);
	void drive(const float & gasStrength, const float & brakeStrength, bool goForward);
	void parkingBrake();
	
	void differential(float * dst) const;
	void wheelForce(float * dst) const;
	void wheelSlip(float * dst) const;
	void wheelSkid(float * dst) const;
	
	static float RodRadius;
	static btRigidBody * ChassisBody;
	static Vector3F ChassisOrigin;
	static int Gear;

private:
	btRigidBody* createCarrier(const Matrix44F & tm, bool isLeft);
	btRigidBody* createWishbone(btRigidBody* carrier, const Matrix44F & tm, bool isUpper, bool isLeft);
	btRigidBody* createSteeringArm(btRigidBody* carrier, const Matrix44F & tm, bool isLeft);
	btRigidBody* createDamper(btRigidBody * lowerArm, const Matrix44F & tm, bool isLeft);
	btRigidBody* createSwayBar(const Matrix44F & tm, btRigidBody * arm, bool isLeft);
	void connectSwayBar(const Matrix44F & tm, btRigidBody * bar);
	btCompoundShape* createWishboneShape(bool isUpper, bool isLeft);
	const Matrix44F wishboneHingTMLocal(bool isUpper, bool isLeft, bool isFront) const;
	void wishboneLA(bool isUpper, bool isLeft, bool isFront, float & l, float & a) const;
	void connectArm(btRigidBody* arm, const Matrix44F & tm, bool isUpper, bool isLeft, bool isFront);
	void steerWheel(const float & ang, int i);
	void releaseBrake();
	void applyMotor(float rps, const int & i, float force);
	void brake(const int & i, const float & strength, bool goForward);
	const Matrix44F wheelHubTM(const int & i) const;
	const Vector3F wheelVelocity(const int & i) const;
	float limitDrive(const int & i, const float & targetSpeed, bool goForward = true);
	void brake(const float & strength, bool goForward);
	void power(const float & strength, bool goForward);
	void power(const int & i, const float & strength, bool goForward);
	const float wheelSlip(const int & i) const;
	const float wheelSkid(const int & i) const;
	Profile m_profile;
	btGeneric6DofConstraint* m_steerJoint[2];
	btGeneric6DofConstraint* m_driveJoint[2];
	btRigidBody * m_wheelHub[2];
	Wheel * m_wheel[2];
	btRigidBody * m_swayBarLeft;
	Damper * m_damper[2];
	float m_differential[2];
	float m_wheelForce[2];
};
}