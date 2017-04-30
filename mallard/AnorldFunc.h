#pragma once
#include <RenderEngine.h>
#ifdef WIN32
#include <ai.h>
#endif
#include <AllMath.h>
class AnorldFunc : public RenderEngine{
public:
    AnorldFunc();
    virtual ~AnorldFunc();
    
    virtual void render();
    void logArnoldVersion() const;
    void loadPlugin(const char * fileName);
	std::string arnoldVersionString();
#ifdef WIN32    
    void logAStrArray(AtArray *arr);
    void logAMatrix(AtMatrix matrix);
    void logRenderError(int status);
    void setMatrix(const Matrix44F & src, AtMatrix & dst) const;
#endif
protected:

private:
};
