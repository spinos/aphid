#ifndef DYNAMICBODY_H
#define DYNAMICBODY_H

namespace aphid {
class DynamicBody 
{
public:
    DynamicBody();
    virtual ~DynamicBody();
    
    bool isSleeping() const;
    void putToSleep();
    void wakeUp();
protected:
    virtual void stopMoving();
private:
    bool m_isSleeping;
};

}
#endif        //  #ifndef DYNAMICBODY_H

