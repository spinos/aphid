#pragma once
#include "IVelocityFile.h"

class OVelocityFile : public IVelocityFile {
public:
    OVelocityFile();
    OVelocityFile(const char * name);
    virtual ~OVelocityFile();
    
    virtual void createPoints(unsigned n);
	void setCurrentTranslation(const Vector3F & t);
	void setCurrentP(const Vector3F * src, unsigned nv, unsigned nvdrift);
	bool writeFrameTranslationalVelocity();
    bool writeFrameVelocity();
	bool writeMaxSpeed();
	
protected:
    bool writeNumPoints(int n);
    void calculateVelocity();
	void zeroVelocity();
	void writeTranslationalVelocity(int t);
    void writeVelocity(int t);
private:
	Vector3F m_currentT, m_lastT;
    BaseBuffer * m_currentP;
    BaseBuffer * m_lastP;
};
