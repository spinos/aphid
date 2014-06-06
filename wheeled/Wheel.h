/*
 *  Wheel.h
 *  wheeled
 *
 *  Created by jian zhang on 5/31/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "Tire.h"
#include <Common.h>
namespace caterpillar {
class Wheel {
public:
	struct Profile {
		Profile() {
			_width = 2.67f;
			_radiusMajor = 4.57f;
			_radiusMinor = .312f;
			_mass = 1.f;
		}
		float _width, _radiusMajor, _radiusMinor, _mass;
	};
	
	Wheel();
	virtual ~Wheel();
	
	void setProfile(const Profile & info);
	const float width() const;
	const float radius() const;
	
	btRigidBody* create(const Matrix44F & tm, bool isLeft);
	btRigidBody* body();
	const Vector3F velocity() const;
	const btTransform tm() const;
private:
	Profile m_profile;
	Tire m_tire;
	btRigidBody * m_body;
};
}