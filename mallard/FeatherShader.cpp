#include "FeatherShader.h"

FeatherShader::FeatherShader() 
{ 
    setShaderType(TFeather);
    m_gloss = 10.f;
    m_gloss2 = 7.f;
}

void FeatherShader::setDiffuseMapName(const std::string & name) { m_diffuseMapName = name; }
void FeatherShader::setOpacityMapName(const std::string & name) { m_opacityMapName = name; }
void FeatherShader::setSpecularMapName(const std::string & name) { m_specularMapName = name; }
void FeatherShader::setSpecular2MapName(const std::string & name) { m_specular2MapName = name; }
void FeatherShader::setGloss(float x) { m_gloss = x; }
void FeatherShader::setGloss2(float x) { m_gloss2 = x; }

float FeatherShader::gloss() const { return m_gloss; }
float FeatherShader::gloss2() const { return m_gloss2; }
