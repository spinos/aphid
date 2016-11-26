#ifndef APH_GLSL_INSTANCER_H
#define APH_GLSL_INSTANCER_H

#include "GlslBase.h"

namespace aphid {

class GlslInstancer : public GLSLBase 
{
public:
    GlslInstancer();
    virtual ~GlslInstancer();
    
protected:
    virtual const char* vertexProgramSource() const;
	virtual const char* fragmentProgramSource() const;
	
private:
    
};

}
#endif
