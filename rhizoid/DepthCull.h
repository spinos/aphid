/*
 *  depthCut.h
 *  proxyPaint
 *
 *  Created by jian zhang on 5/18/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include <GlslBase.h>
#include <vector>
class ATriangleMesh;
class DepthCull : public GLSLBase
{
public:
	DepthCull();
	virtual ~DepthCull();
	
	virtual const char* vertexProgramSource() const;
	virtual const char* fragmentProgramSource() const;
	void drawFrameBuffer(const std::vector<ATriangleMesh *> & meshes);
	
	void setLocalSpace(double* m);
	
	char isCulled(float depth, int x, int y, int w, int h, float offset);
	
private:
	double *fLocalSpace;

};