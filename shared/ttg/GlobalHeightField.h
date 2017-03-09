/*
 *  GlobalHeightField.h
 *  
 *
 *  Created by jian zhang on 3/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TTG_GLOBAL_HEIGHT_FIELD_H
#define APH_TTG_GLOBAL_HEIGHT_FIELD_H

#include <math/Vector3F.h>

namespace aphid {

namespace ttg {

class GlobalHeightField {

	Vector3F m_planetCenter;
	
public:
	GlobalHeightField();
	
	void setPlanetRadius(float x);
	float sample(const Vector3F & pos) const;
	
};

}

}
#endif