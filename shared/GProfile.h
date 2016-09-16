/*
 *  GProfile.h
 *  mallard
 *
 *  Created by jian zhang on 9/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APHID_G_PROFILE_H
#define APHID_G_PROFILE_H

#include <GMaterial.h>
#include <GLight.h>

namespace aphid {

class GProfile {
public:
	GProfile();
		
	GProfile(bool lighting, bool depthTest, bool wired, bool culled, bool textured);	
	void apply() const;
	
	GMaterial * m_material;
	bool m_lighting, m_depthTest, m_wired, m_culled, m_textured;
};

}
#endif