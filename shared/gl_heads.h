/*
 *  gl_heads.h
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef APH_GL_HEADS_H
#define APH_GL_HEADS_H

#ifdef WIN32
#include <ogl/gExtension.h>
#endif

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <OpenGL/glext.h>
#include <GLUT/glut.h>
#endif

#ifdef LINUX
#include <GL/glew.h>
#include <GL/gl.h>
#endif

#endif


