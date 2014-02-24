#pragma once
#include <BaseShader.h>
#include <vector>
class ShaderGroup {
public:
	ShaderGroup();
	virtual ~ShaderGroup();
	void addShader(BaseShader * s);
	unsigned numShaders() const;
	BaseShader * getShader(unsigned idx) const;
	BaseShader * getShader(const std::string & name) const;
	void clearShaders();

protected:
	
private:
	std::vector<BaseShader *> m_shaders;
};
