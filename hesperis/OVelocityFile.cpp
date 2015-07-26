#include "OVelocityFile.h"
#include <BaseBuffer.h>
#include <HBase.h>
#include <Vector3F.h>
#include <boost/format.hpp>

OVelocityFile::OVelocityFile() : IVelocityFile() 
{
    m_lastP = new BaseBuffer;
    m_currentP = new BaseBuffer;
}

OVelocityFile::OVelocityFile(const char * name) : IVelocityFile(name) 
{
    m_lastP = new BaseBuffer;
    m_currentP = new BaseBuffer;
}

OVelocityFile::~OVelocityFile() 
{
    delete m_currentP;
    delete m_lastP;
}

void OVelocityFile::createPoints(unsigned n)
{ 
    IVelocityFile::createPoints(n);
    m_currentP->create(n*12);
    m_lastP->create(n*12);
    writeNumPoints(n);
}

void OVelocityFile::setCurrentP(const Vector3F * src, unsigned nv, unsigned nvdrift)
{ m_currentP->copyFrom(src, nv*12, nvdrift*12); }

bool OVelocityFile::writeNumPoints(int n)
{
    useDocument();
    HBase vg("/vel");
    if(!vg.hasNamedAttr(".nv"))
        vg.addIntAttr(".nv");
    vg.writeIntAttr(".nv", &n);
    vg.close();
    return true;
}

bool OVelocityFile::writeFrameVelocity()
{
    if(!isFrameBegin()) {
        calculateVelocity();
        writeVelocity(currentFrame()-1);
        if(isFrameEnd()) {
			zeroVelocity();
            writeVelocity(currentFrame());
		}
    }
    m_lastP->copyFrom(m_currentP->data(), numPoints() * 12);
    return true;
}

void OVelocityFile::calculateVelocity()
{
    Vector3F * pos0 = (Vector3F *)m_lastP->data();
    Vector3F * pos1 = (Vector3F *)m_currentP->data();
    unsigned i=0;
    for(;i<numPoints();i++)
        velocities()[i] = (pos1[i] - pos0[i]) * 30.f; // 30 fps
}

void OVelocityFile::zeroVelocity()
{
	unsigned i=0;
    for(;i<numPoints();i++)
        velocities()[i] = Vector3F::Zero;
}

void OVelocityFile::writeVelocity(int t)
{
    std::cout<<"\n write velocity at frame "<<t;
    useDocument();
    HBase vg("/vel");
    const std::string sframe = boost::str( boost::format("%1%") % t );
    if(!vg.hasNamedData(sframe.c_str()))
	    vg.addVector3Data(sframe.c_str(), numPoints());
    vg.writeVector3Data(sframe.c_str(), numPoints(), velocities());
    vg.close();
}
//:~
