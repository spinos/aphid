#pragma once
#include "BaseShader.h"

class FeatherShader : public BaseShader {
public:
    FeatherShader();
    void setDiffuseMapName(const std::string & name);
    void setOpacityMapName(const std::string & name);
    void setSpecularMapName(const std::string & name);
    void setSpecular2MapName(const std::string & name);
    void setGloss(float x);
    void setGloss2(float x);
    
    float gloss() const;
    float gloss2() const;
protected:

private:
    std::string m_diffuseMapName;
    std::string m_opacityMapName;
    std::string m_specularMapName;
    std::string m_specular2MapName;
    float m_gloss, m_gloss2;
};
