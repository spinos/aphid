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
	bool readMaxSpeed();

    Vector3F * velocities() const;
    BaseBuffer * velocityBuf() const;
	
	Vector3F * translationalVelocity();
	
	void resetMaxSpeed();
	void updateMaxSpeed(float x);
	float maxSpeed() const;
protected:
	const std::string translationName() const;
	const std::string deformationName() const;
	const std::string maxSpeedName() const;
private:
	Vector3F m_translationalVelocity;
    BaseBuffer * m_vel;
	unsigned m_numV;
	float m_maxSpeed;
};
#endif        //  #ifndef IVELOCITYFILE_H

