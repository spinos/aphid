#ifndef IVELOCITYFILE_H
#define IVELOCITYFILE_H

#include <APlaybackFile.h>
class BaseBuffer;
class IVelocityFile : public APlaybackFile {
public:
    IVelocityFile();
    IVelocityFile(const char * name);
    virtual ~IVelocityFile();
    
    virtual void createPoints(unsigned n);
    const unsigned numPoints() const;
    
    unsigned readNumPoints();
    bool readFrameVelocity();

    Vector3F * velocities() const;
    BaseBuffer * velocityBuf() const;
protected:   
private:
    BaseBuffer * m_vel;
    unsigned m_numV;
};
#endif        //  #ifndef IVELOCITYFILE_H

