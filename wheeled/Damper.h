/*
 *  Damper.h
 *  wheeled
 *
 *  Created by jian zhang on 6/6/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Common.h>
namespace caterpillar {
class Damper  {
public:
	Damper(btRigidBody * lower, btRigidBody * upper, const float & l);
	virtual ~Damper();
	void update();
protected:
	const float relativeSpeed() const;
	const bool isCompressing() const;
private:
	btRigidBody * m_body[2];
	btGeneric6DofSpringConstraint* m_slid;
	float m_range;
};
}