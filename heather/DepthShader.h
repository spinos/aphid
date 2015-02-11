/*
 *  DepthShader.h
 *  heather
 *
 *  Created by jian zhang on 2/12/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <GlslBase.h>

class DepthShader : public GLSLBase {
public:
	DepthShader();
	virtual ~DepthShader();
protected:
	virtual const char* vertexProgramSource() const;
	virtual const char* fragmentProgramSource() const;
	virtual void updateShaderParameters() const;
private:
};