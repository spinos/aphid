/*
 *  glslBase.h
 *  
 *
 *  Created by jian zhang on 10/31/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <string>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <OpenGL/glext.h>
#include <GLUT/glut.h>
#endif

#ifdef _WIN32
#include "../shared/gExtension.h"
#endif

#ifdef LINUX
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>
#include <X11/Intrinsic.h>
#endif

class GLSLBase
{
public:
	GLSLBase();
	virtual ~GLSLBase();
	
	char diagnose(std::string& log);
	char initializeShaders(std::string& log);
	char initializeFBO(std::string& log);
	
	char isDiagnosed() const { return fHasDiagnosis; }
	char hasFBO() const { return fHasFBO; }
	
	void frameBufferBegin() const;
	void frameBufferEnd() const;
	void showFrameBuffer() const;
	
	void programBegin() const;
	void programEnd() const;
	
	virtual const char* vertexProgramSource() const;
	virtual const char* fragmentProgramSource() const;
	virtual void defaultShaderParameters();
	virtual void updateShaderParameters() const;
	virtual int frameBufferWidth() const;
	virtual int frameBufferHeight() const;
	virtual void drawFrameBuffer();
	
	char fHasDiagnosis, fHasExtensions, fHasFBO;
	GLhandleARB vertex_shader, fragment_shader, program_object;
	GLuint fbo;
	GLuint depthBuffer;
	GLuint img;
	
	float *fPixels;
};
