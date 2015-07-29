#ifndef IVELOCITYFILE_H
#define IVELOCITYFILE_H

#include <APlaybackFile.h>
#include <Vector3F.h>
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
	bool readFrameTranslationalVelocity();

    Vector3F * velocities() const;
    BaseBuffer * velocityBuf() const;
	
	Vector3F * translationalVelocity();
protected:
	const std::string translationName() const;
	const std::string deformationName() const;
private:
	Vector3F m_translationalVelocity;
    BaseBuffer * m_vel;
    unsigned m_numV;
};
#endif        //  #ifndef IVELOCITYFILE_H

