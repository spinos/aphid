#include "BaseShader.h"

BaseShader::BaseShader() { setEntityType(TShader); }

void BaseShader::setShaderType(BaseShader::ShaderType t) { m_shaderType = t; }
BaseShader::ShaderType BaseShader::shaderType() const { return m_shaderType; }
