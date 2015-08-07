#ifndef AWORLD_H
#define AWORLD_H
#include <string>
class AWorld
{
public:
    AWorld();
    virtual ~AWorld();
    
    virtual void stepPhysics(float dt);
	virtual void prePhysics();
	virtual void postPhysics();
    virtual void progressFrame();
protected:

private:

};
#endif        //  #ifndef AWORLD_H

