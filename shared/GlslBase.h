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
	char m_hasShaders, fHasFBO;
	
	GLhandleARB vertex_shader, fragment_shader, program_object;
	GLuint fbo;
	GLuint depthBuffer;
	GLuint img;
	float *fPixels;
	
	static float CoreVersion;
	
public:
	GLSLBase();
	virtual ~GLSLBase();
	
	static bool diagnose(std::string& log);
	char initializeShaders(std::string& log);
	char initializeFBO(std::string& log);
	
	bool isDiagnosed() const;
	char hasFBO() const;
	
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
	virtual void defaultShaderParameters();
	virtual void updateShaderParameters() const;
	
	GLhandleARB * program();
	
	const float * pixels() const;
	
};
}
#endif        //  #ifndef GLSLBASE_H

