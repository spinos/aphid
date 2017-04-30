/*
 *  GlobalElevation.h
 *  
 *  holding a number of height fields
 *  defaul elevation <- distance_to_planet_center - planet_radius
 *  
 *  Created by jian zhang on 3/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TTG_GLOBAL_ELEVATION_H
#define APH_TTG_GLOBAL_ELEVATION_H

#include <math/Vector3F.h>
#include <vector>
#include <string>

namespace aphid {

namespace img {

class HeightField;

}

namespace ttg {

class GlobalElevation {

typedef std::vector<img::HeightField * > FieldVecTyp;
	FieldVecTyp m_fields;
	Vector3F m_planetCenter;
	
public:
	GlobalElevation();
	virtual ~GlobalElevation();
	
	bool loadHeightField(const std::string & fileName);
	
	void setPlanetRadius(float x);
	float sample(const Vector3F & pos) const;
	int numHeightFields() const;
	const img::HeightField & heightField(int i) const;
	
protected:

private:
	void internalClear();
	
};

}

}
#endif