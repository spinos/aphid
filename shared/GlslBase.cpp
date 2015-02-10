/*
 *  GLSLBase.cpp
 *  pmap
 *
 *  Created by jian zhang on 10/31/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include "GlslBase.h"




typedef struct glExtensionEntry {
    char* name;
    GLfloat promoted;
    GLboolean supported;
} glExtensionEntry;

glExtensionEntry entriesNeeded[] = {
{"GL_EXT_framebuffer_object",   0.0, 0},
{"GL_ARB_texture_cube_map",     1.3, 0},
{"GL_ARB_shader_objects",       2.0, 0},
{"GL_ARB_shading_language_100", 2.0, 0},
{"GL_ARB_fragment_shader",      2.0, 0},
{"GL_ARB_vertex_buffer_object",      2.0, 0},
{"GL_ARB_multitexture",      1.3, 0},
{"GL_ARB_multisample",      1.3, 0},
{"GL_ARB_vertex_program",      2.0, 0},
{"GL_ARB_fragment_program",      2.0, 0},
{"GL_ARB_texture_rectangle",      0.0, 0},
};

GLSLBase::GLSLBase() : fHasDiagnosis(0), fHasExtensions(0) {}

GLSLBase::~GLSLBase() 
{
	if (program_object) 
		glDeleteObjectARB(program_object);
}

const char* GLSLBase::vertexProgramSource() const
{
	return "void main()"
"{"
"		gl_Position = ftransform();"
"		gl_FrontColor = gl_Color;"
"}";
}

const char* GLSLBase::fragmentProgramSource() const
{
	return "void main()"
"{"
"		gl_FragColor = gl_Color * vec4 (0.5);"
"}";
}

/*
void GLSLBase::start(Z3DTexture* db) const
{
	glBindTexture(GL_TEXTURE_3D, noisepool);
	glUseProgramObjectARB(program_object);	
// set global values
	glUniform1fARB(glGetUniformLocationARB(program_object, "KNoise"), fKNoise);
	glUniform3fARB(glGetUniformLocationARB(program_object, "LightPosition"), fLightPos.x, fLightPos.y, fLightPos.z);
	glUniform1fARB(glGetUniformLocationARB(program_object, "KDiffuse"), fKDiffuse);
	glUniform1fARB(glGetUniformLocationARB(program_object, "GScale"), fDensity);
	glUniform3fARB(glGetUniformLocationARB(program_object, "CCloud"), fCCloud.x, fCCloud.y, fCCloud.z);
	glUniform1fARB(glGetUniformLocationARB(program_object, "Lacunarity"), fLacunarity);
	glUniform1fARB(glGetUniformLocationARB(program_object, "Dimension"), fDimension);
	
	db->setProgram(program_object);
}

void GLSLBase::end() const
{
	glUseProgramObjectARB(NULL);
}
*/

#include <sstream>
char GLSLBase::diagnose(std::string& log)
{
	float core_version;
	// sscanf((char *)glGetString(GL_VERSION), "%f", &core_version);
	std::stringstream sst;
	sst.str((char *)glGetString(GL_VERSION));
	sst>>core_version;
	
	//char sbuf[64];
	//sprintf(sbuf, "%s version %s\n", (char *)glGetString(GL_RENDERER), (char *)glGetString(GL_VERSION));
	//log = sbuf;

	std::stringstream sst1;
	sst1.str("");
	sst1<<(char *)glGetString(GL_RENDERER)<<" version "<<(char *)glGetString(GL_VERSION);
	log = sst.str();
	

	int supported = 1;
	int j = sizeof(entriesNeeded)/sizeof(glExtensionEntry);
	
#ifdef WIN32
     char sbuf[64];
	 for (int i = 0; i < j; i++) {
		 if(!gCheckExtension(entriesNeeded[i].name)) {
			 sprintf(sbuf, "%-32s %d\n", entriesNeeded[i].name, 0);
			 supported = 0;
		 }
		else sprintf(sbuf, "%-32s %d\n", entriesNeeded[i].name, 1);
		log += sbuf;
	}
#else
	const GLubyte *strExt = glGetString(GL_EXTENSIONS);
	for (int i = 0; i < j; i++) {
		entriesNeeded[i].supported = gluCheckExtension((GLubyte*)entriesNeeded[i].name, strExt) |
		(entriesNeeded[i].promoted && (core_version >= entriesNeeded[i].promoted));
		// sprintf(sbuf, "%-32s %d\n", entriesNeeded[i].name, entriesNeeded[i].supported);
		sst1.str("");
		sst1<<entriesNeeded[i].name<<" "<<entriesNeeded[i].supported;
		log += sst1.str();
		supported &= entriesNeeded[i].supported;
	}
#endif	
	if(core_version < 1.4) {
		log += "OpenGL version too low, this thing may not work correctly!\n";
	}
	
	if(supported != 1) return 0;

	fHasExtensions = initializeShaders(log);
		
	fHasDiagnosis = 1;
	return 1;
}

char GLSLBase::initializeShaders(std::string& log)
{
	GLint vertex_compiled, fragment_compiled;
	GLint linked;
		
// Delete any existing program object 
	if (program_object) {
		glDeleteObjectARB(program_object);
		program_object = NULL;
	}
		
	vertex_shader = glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);
	const char* a = vertexProgramSource();
	glShaderSourceARB(vertex_shader, 1, &a, NULL);
	glCompileShaderARB(vertex_shader);
	glGetObjectParameterivARB(vertex_shader, GL_OBJECT_COMPILE_STATUS_ARB, &vertex_compiled);
	
	fragment_shader   = glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);
	const char* b = fragmentProgramSource();
	glShaderSourceARB(fragment_shader, 1, &b, NULL);
	glCompileShaderARB(fragment_shader);
	glGetObjectParameterivARB(fragment_shader, GL_OBJECT_COMPILE_STATUS_ARB, &fragment_compiled);
	
	if (!vertex_compiled || !fragment_compiled) {
		if (vertex_shader) {
			glDeleteObjectARB(vertex_shader);
			vertex_shader   = NULL;
		}
		if (fragment_shader) {
			glDeleteObjectARB(fragment_shader);
			fragment_shader = NULL;
		}
		log += "\nshaders not compiled";
		return 0;
	}
	
	program_object = glCreateProgramObjectARB();
	if (vertex_shader != NULL)
	{
		glAttachObjectARB(program_object, vertex_shader);
		glDeleteObjectARB(vertex_shader);
	}
	if (fragment_shader != NULL)
	{
		glAttachObjectARB(program_object, fragment_shader);
		glDeleteObjectARB(fragment_shader);
	}
	glLinkProgramARB(program_object);
	glGetObjectParameterivARB(program_object, GL_OBJECT_LINK_STATUS_ARB, &linked);
		
	if (!linked) {
		glDeleteObjectARB(program_object);
		program_object = NULL;
		log += "shaders not linked";
		return 0;
	}

	defaultShaderParameters();
	return 1;
}

void GLSLBase::defaultShaderParameters()
{
	/*glUseProgramObjectARB(program_object);
	
	glUniform1iARB(glGetUniformLocationARB(program_object, "EarthNight"), 0);
	glUniform1fARB(glGetUniformLocationARB(program_object, "KNoise"), fKNoise);
	glUniform1fARB(glGetUniformLocationARB(program_object, "KDiffuse"), fKDiffuse);
	glUniform3fARB(glGetUniformLocationARB(program_object, "LightPosition"), fLightPos.x, fLightPos.y, fLightPos.z);

	glUseProgramObjectARB(NULL);*/
}

void GLSLBase::updateShaderParameters() const
{

}

void GLSLBase::programBegin() const
{
	glUseProgramObjectARB(program_object);
	updateShaderParameters();
}

void GLSLBase::programEnd() const
{
	glUseProgramObjectARB(NULL);
}
