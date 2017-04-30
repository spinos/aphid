#include "BaseShader.h"

BaseShader::BaseShader() {}

void BaseShader::setShaderType(BaseShader::ShaderType t) { m_shaderType = t; }
BaseShader::ShaderType BaseShader::shaderType() const { return m_shaderType; }

const TypedEntity::Type BaseShader::type() const
{ return TShader; }
