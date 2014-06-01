/*
 *  Wheel.h
 *  wheeled
 *
 *  Created by jian zhang on 5/31/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include "TireMesh.h"
namespace caterpillar {
class Wheel {
public:
	struct Profile {
		Profile() {
			_width = 2.67f;
			_radiusMajor = 4.57f;
			_radiusMinor = .312f;
		}
		float _width, _radiusMajor, _radiusMinor;
	};
	
	Wheel();
	virtual ~Wheel();
	
	void setProfile(const Profile & info);
	const float width() const;
	
	void createShape();
	btRigidBody* create(const Matrix44F & tm);
private:
	Profile m_profile;
	TireMesh m_tire;
	btCollisionShape* m_shape;
};
}