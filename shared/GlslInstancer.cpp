/*
 * https://www.opengl.org/wiki/Vertex_Rendering#Instancing
 * http://www.informit.com/articles/article.aspx?p=2033340&seqNum=5
 * https://www.khronos.org/opengles/sdk/tools/Reference-Compiler/
 * http://renderingpipeline.com/2012/03/shader-model-and-glsl-versions/
 * Legacy Instancer (GL core < 3.3) use glsl_pseudo_instancing
 */
#include "GlslInstancer.h"

namespace aphid {
    
GlslLegacyInstancer::GlslLegacyInstancer()
{
    m_distantLightVec.set(1, 1, 1);
    m_distantLightVec.normalize();
}

GlslLegacyInstancer::~GlslLegacyInstancer()
{}

const char* GlslLegacyInstancer::vertexProgramSource() const
{
	return "#version 120\n"
"uniform mat4 worldMatrix;"
"varying vec3 shadingNormal;"
"void main()"
"{"
"   vec4 positionWorld;"
"   vec4 worldViewPosition;"
"   vec3 normalWorld;"
"   positionWorld.x = dot(gl_MultiTexCoord1, gl_Vertex);"
"   positionWorld.y = dot(gl_MultiTexCoord2, gl_Vertex);"
"   positionWorld.z = dot(gl_MultiTexCoord3, gl_Vertex);"
"   positionWorld.w = 1.0;"
"   worldViewPosition = worldMatrix * positionWorld;"
//"   gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix *  worldViewPosition;"
"   gl_Position = gl_ModelViewProjectionMatrix * worldViewPosition;"
"   normalWorld.x = dot(gl_MultiTexCoord1.xyz, gl_Normal);"
"   normalWorld.y = dot(gl_MultiTexCoord2.xyz, gl_Normal);"
"   normalWorld.z = dot(gl_MultiTexCoord3.xyz, gl_Normal);"
"   normalWorld = normalize(normalWorld);"
"   shadingNormal = normalWorld;"
"}";
}

const char* GlslLegacyInstancer::fragmentProgramSource() const
{
	return "#version 120\n"
"uniform vec3 distantLightVec;"
"uniform vec3 diffuseColor;"
"varying vec3 shadingNormal;"
"void main()"
"{"
"   float ldn = dot(shadingNormal, distantLightVec);"
"   if(ldn < 0.0) ldn = 0.0;\n"
"	gl_FragColor = vec4(diffuseColor * ldn, 1.0);"
"}";
}

void GlslLegacyInstancer::defaultShaderParameters()
{
    m_worldMatLoc = glGetUniformLocationARB(*program(), "worldMatrix");
    m_distantLightVecLoc = glGetUniformLocationARB(*program(), "distantLightVec");
	m_diffColorLoc = glGetUniformLocationARB(*program(), "diffuseColor");
}

void GlslLegacyInstancer::updateShaderParameters() const
{
    glUniformMatrix4fvARB(m_worldMatLoc, 1, 0, (float*)&m_worldMat);
    glUniform3fvARB(m_distantLightVecLoc, 1, (float*)&m_distantLightVec);
}

void GlslLegacyInstancer::setWorldTm(const Matrix44F & x)
{ m_worldMat = x; }

void GlslLegacyInstancer::setDistantLightVec(const Vector3F & x)
{ m_distantLightVec = x.normal(); }

void GlslLegacyInstancer::setDiffueColorVec(const float * x)
{ glUniform3fvARB(m_diffColorLoc, 1, x); }


GlslLegacyFlatInstancer::GlslLegacyFlatInstancer()
{}

GlslLegacyFlatInstancer::~GlslLegacyFlatInstancer()
{}

const char* GlslLegacyFlatInstancer::vertexProgramSource() const
{
	return "#version 120\n"
"uniform mat4 worldMatrix;"
"void main()"
"{"
"   vec4 positionWorld;"
"   vec4 worldViewPosition;"
"   positionWorld.x = dot(gl_MultiTexCoord1, gl_Vertex);"
"   positionWorld.y = dot(gl_MultiTexCoord2, gl_Vertex);"
"   positionWorld.z = dot(gl_MultiTexCoord3, gl_Vertex);"
"   positionWorld.w = 1.0;"
"   worldViewPosition = worldMatrix * positionWorld;"
"   gl_Position = gl_ModelViewProjectionMatrix * worldViewPosition;"
"}";
}

const char* GlslLegacyFlatInstancer::fragmentProgramSource() const
{
	return "#version 120\n"
"uniform vec3 diffuseColor;"
"void main()"
"{"
"	gl_FragColor = vec4(diffuseColor, 1.0);"
"}";
}

void GlslLegacyFlatInstancer::defaultShaderParameters()
{
    m_worldMatLoc = glGetUniformLocationARB(*program(), "worldMatrix");
    m_diffColorLoc = glGetUniformLocationARB(*program(), "diffuseColor");
}

void GlslLegacyFlatInstancer::updateShaderParameters() const
{
    glUniformMatrix4fvARB(m_worldMatLoc, 1, 0, (float*)&m_worldMat);
}

void GlslLegacyFlatInstancer::setWorldTm(const Matrix44F & x)
{ m_worldMat = x; }

void GlslLegacyFlatInstancer::setDiffueColorVec(const float * x)
{ glUniform3fvARB(m_diffColorLoc, 1, x); }


GlslInstancer::GlslInstancer()
{}

GlslInstancer::~GlslInstancer()
{}

const char* GlslInstancer::vertexProgramSource() const
{
	return "#version 330\n"
"uniform mat4 ModelViewMatrix;"
"uniform mat4 ProjectionMatrix;"
// "layout(location = 0) in vec3 in_position;"
"void main()"
"{"
// "		gl_Position = ftransform();"
"   gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;"
"		gl_FrontColor = gl_Color;"
"}";
}

const char* GlslInstancer::fragmentProgramSource() const
{
	return "#version 330\n"
"void main()"
"{"
"		gl_FragColor = gl_Color * vec4 (0.99);"
"}";
}

}
