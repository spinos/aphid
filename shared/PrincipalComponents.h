/*
 *  PrincipalComponents.h
 *  aphid
 *
 *  Created by jian zhang on 8/22/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <AllMath.h>
class PrincipalComponents {
public:
	PrincipalComponents();
	virtual ~PrincipalComponents();
	
	void analyze(Vector3F * pos, unsigned n);
	
protected:
	float covarianceXX(Vector3F * pos, unsigned n) const;
	float covarianceXY(Vector3F * pos, unsigned n) const;
	float covarianceXZ(Vector3F * pos, unsigned n) const;
	float covarianceYX(Vector3F * pos, unsigned n) const;
	float covarianceYY(Vector3F * pos, unsigned n) const;
	float covarianceYZ(Vector3F * pos, unsigned n) const;
	float covarianceZX(Vector3F * pos, unsigned n) const;
	float covarianceZY(Vector3F * pos, unsigned n) const;
	float covarianceZZ(Vector3F * pos, unsigned n) const;
private:
	
};