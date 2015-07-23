#pragma once
#include <APlaybackFile.h>
class BaseBuffer;
class IVelocityFile : public APlaybackFile {
public:
    IVelocityFile();
    IVelocityFile(const char * name);
    virtual ~IVelocityFile();
    
    virtual void createPoints(unsigned n);
    const unsigned numPoints() const;
protected:
    Vector3F * velocities() const;
private:
    BaseBuffer * m_vel;
    unsigned m_numV;
};
