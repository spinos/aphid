#include "DynamicBody.h"
namespace aphid {
    
DynamicBody::DynamicBody() 
{ m_isSleeping = false; }

DynamicBody::~DynamicBody() {}

bool DynamicBody::isSleeping() const
{ return m_isSleeping; }
    
void DynamicBody::putToSleep()
{ 
    m_isSleeping = true; 
    stopMoving();
}

void DynamicBody::wakeUp()
{ m_isSleeping = false; }

void DynamicBody::stopMoving() {}

}
