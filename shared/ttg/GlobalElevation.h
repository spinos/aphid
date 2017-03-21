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
	static FieldVecTyp m_fields;
	Vector3F m_planetCenter;
	
public:
	GlobalElevation();
	virtual ~GlobalElevation();
	
	static bool LoadHeightField(const std::string & fileName);
	static std::string LastFileBaseName();
	static int NumHeightFields();
	static const img::HeightField & GetHeightField(int i);
	static img::HeightField * HeightFieldR(int i);
	
	void setPlanetRadius(float x);
	float sample(const Vector3F & pos) const;
	
protected:

private:
	void internalClear();
	
};

}

}
#endif