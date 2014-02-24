#include "MlShaders.h"
#include "FeatherShader.h"
MlShaders::MlShaders() 
{
    FeatherShader * s = new FeatherShader;
    s->setName("feather");
    addShader(s);
}

MlShaders::~MlShaders() {}
