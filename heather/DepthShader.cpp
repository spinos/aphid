/*
 *  DepthShader.cpp
 *  heather
 *
 *  Created by jian zhang on 2/12/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "DepthShader.h"

DepthShader::DepthShader() {}
DepthShader::~DepthShader() {}

const char* DepthShader::vertexProgramSource() const
{
	return "varying float distToCamera;"
"void main()"
"{"
"		gl_Position = ftransform();"
"	vec4 cs_position = gl_ModelViewMatrix * gl_Vertex;"
"	distToCamera = -cs_position.z;"
"}";
}

const char* DepthShader::fragmentProgramSource() const
{
	return  "varying float distToCamera;"
"void main()"
"{"
"	float d = distToCamera; "
"	gl_FragColor = vec4 (d, d, d, 1.0);"
"}";
}

void DepthShader::updateShaderParameters() const {}
