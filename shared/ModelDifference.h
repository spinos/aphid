/*
 *  ModelDifference.h
 *  testadenium
 *
 *  Created by jian zhang on 7/14/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <vector>
#include <Vector3F.h>
class BaseBuffer;
class AGenericMesh;
class ModelDifference {
public:
	ModelDifference(AGenericMesh * target);
	virtual ~ModelDifference();
	
	bool matchTarget(AGenericMesh * object) const;
	Vector3F resetTranslation(const AGenericMesh * object);
	Vector3F addTranslation(const AGenericMesh * object);
	void computeVelocities(Vector3F * dst, AGenericMesh * object, float oneOverDt);
	
	const unsigned numTranslations() const;
	const Vector3F getTranslation(unsigned idx) const;
    const Vector3F lastTranslation() const;
protected:
	AGenericMesh * target() const;
private:
	std::vector<Vector3F> m_centers;
	BaseBuffer * m_p0;
	AGenericMesh * m_target;
};