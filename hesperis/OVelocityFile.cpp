#include "OVelocityFile.h"
#include <BaseBuffer.h>
#include <HBase.h>
OVelocityFile::OVelocityFile() : APlaybackFile() 
{
    m_vel = new BaseBuffer;
    m_lastP = new BaseBuffer;
    m_currentP = new BaseBuffer;
}

OVelocityFile::OVelocityFile(const char * name) : APlaybackFile(name) 
{
    m_vel = new BaseBuffer;
    m_lastP = new BaseBuffer;
    m_currentP = new BaseBuffer;
}

OVelocityFile::~OVelocityFile() 
{
    delete m_vel;
    delete m_currentP;
    delete m_lastP;
}

void OVelocityFile::createPoints(unsigned n)
{ 
    m_vel->create(n*12);
    m_currentP->create(n*12);
    m_lastP->create(n*12);
    writeNumPoints(n);
}

bool OVelocityFile::writeNumPoints(int n)
{
    useDocument();
    HBase vg("/vel");
    if(vg.hasNamedAttr(".nv"))
        vg.addIntAttr(".nv");
    vg.writeIntAttr(".nv", &n);
    vg.close();
    return true;
}
