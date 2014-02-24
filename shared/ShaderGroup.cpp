#include "ShaderGroup.h"

ShaderGroup::ShaderGroup() {}

ShaderGroup::~ShaderGroup() { clearShaders(); }

void ShaderGroup::addShader(BaseShader * s)
{
	m_shaders.push_back(s);
}

unsigned ShaderGroup::numShaders() const
{
	return m_shaders.size();
}

BaseShader * ShaderGroup::getShader(unsigned idx) const
{
	return m_shaders[idx];
}

BaseShader * ShaderGroup::getShader(const std::string & name) const
{
	std::vector<BaseShader *>::const_iterator it = m_shaders.begin();
	for(; it != m_shaders.end(); ++it) {
		if((*it)->name() == name)
			return *it;
	}
	return NULL;
}

void ShaderGroup::clearShaders()
{
	std::vector<BaseShader *>::iterator it = m_shaders.begin();
	for(; it != m_shaders.end(); ++it) {
		delete *it;
	}
	m_shaders.clear();
}

