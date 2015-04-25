#pragma once

#include <TypedEntity.h>
#include <NamedEntity.h>
class BaseShader : public TypedEntity, public NamedEntity {
public:
    enum ShaderType {
        TUnknown = 0,
        THair = 1,
        TFeather = 2
    };
    
	BaseShader();
	
	void setShaderType(ShaderType t);
	ShaderType shaderType() const;
	
	virtual const Type type() const;
	
protected:
	
private:
	ShaderType m_shaderType;
};
