/*
 *  ClampShader.cpp
 *  heather
 *
 *  Created by jian zhang on 2/12/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "ClampShader.h"

ClampShader::ClampShader() {}
ClampShader::~ClampShader() {}

void ClampShader::setTextures(const GLuint & bgdZImg, const GLuint & bgdCImg, 
				const GLuint & depthImg, 
				const GLuint & colorImg)
{
	m_bgdZImg = bgdZImg;
	m_bgdCImg = bgdCImg;
	m_depthImg = depthImg;
	m_colorImg = colorImg;
}

void ClampShader::setClippings(const float & nearClipping, const float & farClipping)
{
	m_nearClipping = nearClipping;
	m_farClipping = farClipping;
}

const char* ClampShader::vertexProgramSource() const
{
	return "void main()"
"{"
"		gl_Position = ftransform();"
"gl_TexCoord[0] = gl_MultiTexCoord0;"
"gl_TexCoord[1] = gl_MultiTexCoord1;"
"gl_TexCoord[2] = gl_MultiTexCoord2;"
"}";
}

const char* ClampShader::fragmentProgramSource() const
{
	return "uniform sampler2D bgdZ_texture;"
"uniform sampler2D bgdC_texture;"
"uniform sampler2D depth_texture;"
"uniform sampler2D color_texture;"
"uniform float farClipping;"
"void main()"
"{"
"vec4 bgdC = texture2D(bgdC_texture, gl_TexCoord[0].xy);"
"float fogZ = texture2D(depth_texture, gl_TexCoord[2].xy).r;"
"if(fogZ < 0.001) {"
"gl_FragColor = bgdC;"
"return;"
"}"
"float bgdZ = texture2D(bgdZ_texture, gl_TexCoord[1].xy).r;"
"if(fogZ < bgdZ) gl_FragColor = texture2D(color_texture, gl_TexCoord[2].xy);"
"else gl_FragColor = texture2D(bgdC_texture, gl_TexCoord[0].xy);"
//"gl_FragColor = texture2D(color_texture, gl_TexCoord[2].xy);"
//"gl_FragColor = vec4(gl_TexCoord[2].xy, 0, 1);"
//"gl_FragColor = vec4(bgdZ, bgdZ, bgdZ, 1);"
//"gl_FragColor = bgdC;"
// "if(bgdZ<1000) gl_FragColor = vec4(gl_TexCoord[1].x, gl_TexCoord[1].y, 0.0, 1.0);"
// http://tulrich.com/geekstuff/log_depth_buffer.txt
//      (1<<N) * ( a + b / z )
//      N = number of bits of Z precision
//     a = zFar / ( zFar - zNear )
//     b = zFar * zNear / ( zNear - zFar )
//     z = distance from the eye to the object
/*"float a = farClipping/(farClipping - 0.1);"
"float b = farClipping * 0.1 /(farClipping - 0.1);"
"float dep = ( a + b / texture2D(depth_texture, gl_TexCoord[0].xy).r);"
"if(dep >= bgd) gl_FragColor = vec4(bgd, bgd, bgd, 1.0);"
"else gl_FragColor = texture2D(color_texture, gl_TexCoord[0].xy);"*/
"}";
}

void ClampShader::updateShaderParameters() const
{
	glActiveTexture(GL_TEXTURE0);
    glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, m_bgdZImg);
    glUniform1iARB(glGetUniformLocationARB(program_object, "bgdZ_texture"), 0);
	
	glActiveTexture(GL_TEXTURE1);
    glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, m_bgdCImg);
    glUniform1iARB(glGetUniformLocationARB(program_object, "bgdC_texture"), 1);
    
	glActiveTexture(GL_TEXTURE2);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, m_colorImg);
	glUniform1iARB(glGetUniformLocationARB(program_object, "color_texture"), 2);
	
	glActiveTexture(GL_TEXTURE3);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, m_depthImg);
	glUniform1iARB(glGetUniformLocationARB(program_object, "depth_texture"), 3);
	
	glUniform1fARB(glGetUniformLocationARB(program_object, "farClipping"), m_farClipping);
}