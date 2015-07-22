#pragma once
#include <APlaybackFile.h>
class BaseBuffer;
class OVelocityFile : public APlaybackFile {
public:
    OVelocityFile();
    OVelocityFile(const char * name);
    virtual ~OVelocityFile();
    
    void createPoints(unsigned n);
protected:
    bool writeNumPoints(int n);
private:
    BaseBuffer * m_vel;
    BaseBuffer * m_currentP;
    BaseBuffer * m_lastP;
};
