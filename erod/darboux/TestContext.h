/*
 *  darboux
 */
#ifndef TestContext_H
#define TestContext_H

#include <pbd/pbd_common.h>
#include <pbd/ElasticRodContext.h>

namespace aphid {
class Vector3F;
class Matrix44F;
class Quaternion;
}

class TestContext : public aphid::pbd::ElasticRodContext
{
	
public:
    TestContext();
    ~TestContext();
	
	void getMaterialFrames(aphid::Matrix44F& frmA, aphid::Matrix44F& frmB,
			aphid::Vector3F& darboux, aphid::Vector3F* correctVs);
	
	void rotateFrame(const aphid::Quaternion& rot);

protected:
    virtual void stepPhysics(float dt);
	
private:
   
};

#endif        //  #ifndef TestContext_H

