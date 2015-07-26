#pragma once
#include "IVelocityFile.h"
class OVelocityFile : public IVelocityFile {
public:
    OVelocityFile();
    OVelocityFile(const char * name);
    virtual ~OVelocityFile();
    
    virtual void createPoints(unsigned n);
	void setCurrentP(const Vector3F * src, unsigned nv, unsigned nvdrift);
    bool writeFrameVelocity();
protected:
    bool writeNumPoints(int n);
    void calculateVelocity();
	void zeroVelocity();
    void writeVelocity(int t);
private:
    BaseBuffer * m_currentP;
    BaseBuffer * m_lastP;
};
