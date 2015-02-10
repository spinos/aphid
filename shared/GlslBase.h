#ifndef GLSLBASE_H
#define GLSLBASE_H

/*
 *  glslBase.h
 *  
 *
 *  Created by jian zhang on 10/31/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include <gl_heads.h>
#include <string>

class GLSLBase
{
public:
	GLSLBase();
	virtual ~GLSLBase();
	
	char diagnose(std::string& log);
	char initializeShaders(std::string& log);
	char isDiagnosed() const { return fHasDiagnosis; }
	
	void programBegin() const;
	void programEnd() const;
	
	virtual const char* vertexProgramSource() const;
	virtual const char* fragmentProgramSource() const;
	virtual void defaultShaderParameters();
	virtual void updateShaderParameters() const;
	
	char fHasDiagnosis, fHasExtensions;
	GLhandleARB vertex_shader, fragment_shader, program_object;
};
#endif        //  #ifndef GLSLBASE_H

