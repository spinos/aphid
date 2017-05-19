/*
 *  SelectExmpCondition.h
 *  proxyPaint
 *
 *  Created by jian zhang on 5/19/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_RHIZ_SELECT_EXMP_CONDITION_H
#define APH_RHIZ_SELECT_EXMP_CONDITION_H

#include <math/Matrix44F.h>

namespace aphid {

class SelectExmpCondition {

	Matrix44F m_tm;
	Vector3F m_surfNml;
	
public:
	SelectExmpCondition();
	virtual ~SelectExmpCondition();
	
	void setSurfaceNormal(const Vector3F & nml );
	void setTransform(const Matrix44F & tm);
	
	const Matrix44F& transform() const;
	const Vector3F& surfaceNormal() const;
	
protected:

private:
};

}

#endif