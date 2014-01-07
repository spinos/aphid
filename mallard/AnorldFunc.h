#pragma once
#include <RenderEngine.h>
#include <ai.h>
#include <AllMath.h>
class AnorldFunc : public RenderEngine{
public:
    AnorldFunc();
    virtual ~AnorldFunc();
    
    virtual void render();
    
    void loadPlugin(const char * fileName);
    
    void logAStrArray(AtArray *arr);
    void logAMatrix(AtMatrix matrix);
    void logRenderError(int status);
    
    void setMatrix(const Matrix44F & src, AtMatrix & dst) const;
protected:

private:
};
