#pragma once
#include <APlaybackFile.h>
class BaseBuffer;
class OVelocityFile : public APlaybackFile {
public:
    OVelocityFile();
    OVelocityFile(const char * name);
    virtual ~OVelocityFile();
    
    void createPoints(unsigned n);
	void setCurrentP(const Vector3F * src, unsigned nv, unsigned nvdrift);
protected:
    bool writeNumPoints(int n);
private:
    BaseBuffer * m_vel;
    BaseBuffer * m_currentP;
    BaseBuffer * m_lastP;
};
