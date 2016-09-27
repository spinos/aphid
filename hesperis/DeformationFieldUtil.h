#ifndef OPM_DEFORMATION_FIELD_UTIL_H
#define OPM_DEFORMATION_FIELD_UTIL_H

/*
 *  GeometryUtil.h
 *  opium
 *
 *  Created by jian zhang on 3/3/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include "baseUtil.h"
class DeformationFieldUtil : public BaseUtil{
public:
	DeformationFieldUtil();
	
	void dump(const char *filename, 
	            MDagPathArray &active_list);
    
protected:
	
private:

};
#endif        //  #ifndef OPM_DEFORMATION_FIELD_UTIL_H
