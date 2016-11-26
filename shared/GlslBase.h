#ifndef APH_GLSLBASE_H
#define APH_GLSLBASE_H

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
namespace aphid {
    
class GLSLBase
{
	char fHasDiagnosis, fHasExtensions, fHasFBO;
	
	GLhandleARB vertex_shader, fragment_shader, program_object;
	GLuint fbo;
	GLuint depthBuffer;
	GLuint img;
	float *fPixels;
	
public:
	GLSLBase();
	virtual ~GLSLBase();
	
	char diagnose(std::string& log);
	char initializeShaders(std::string& log);
	char initializeFBO(std::string& log);
	
	char isDiagnosed() const { return fHasDiagnosis; }
	char hasFBO() const { return fHasFBO; }
	
	void programBegin() const;
	void programEnd() const;
	
	void frameBufferBegin() const;
	void frameBufferEnd() const;
	void showFrameBuffer() const;
	
	virtual int frameBufferWidth() const;
	virtual int frameBufferHeight() const;
	virtual void drawFrameBuffer();
	
protected:	
	virtual const char* vertexProgramSource() const;
	virtual const char* fragmentProgramSource() const;
	virtual void updateShaderParameters() const;
	virtual void defaultShaderParameters();

	const float * pixels() const;
	
};
}
#endif        //  #ifndef GLSLBASE_H

