/*
 *  DrawInstance.h
 *  proxyPaint
 *
 *  Created by jian zhang on 12/31/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_OGL_DRAW_INSTANCE_H
#define APH_OGL_DRAW_INSTANCE_H

namespace aphid {

class GlslLegacyInstancer;
class GlslLegacyFlatInstancer;

class DrawInstance {

public:
	DrawInstance();
	virtual ~DrawInstance();
	
	static GlslLegacyInstancer * m_instancer;
	static GlslLegacyFlatInstancer * m_wireInstancer;
	
	bool isGlslReady() const;
	bool prepareGlsl();
	
};

}
#endif