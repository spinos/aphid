#include "IVelocityFile.h"
#include <BaseBuffer.h>
#include <boost/format.hpp>

IVelocityFile::IVelocityFile() : APlaybackFile() 
{
    m_vel = new BaseBuffer;
    m_numV = 0;
}

IVelocityFile::IVelocityFile(const char * name) : APlaybackFile(name) 
{
    m_vel = new BaseBuffer;
    m_numV = 0;
}

IVelocityFile::~IVelocityFile() 
{ delete m_vel; }

const unsigned IVelocityFile::numPoints() const
{ return m_numV; }

void IVelocityFile::createPoints(unsigned n)
{ 
    m_vel->create(n*12);
    m_numV = n;
}

Vector3F * IVelocityFile::velocities() const
{ return (Vector3F *)m_vel->data(); }

unsigned IVelocityFile::readNumPoints()
{
    int nv = 0;
    useDocument();
    HBase vg("/vel");
    if(vg.hasNamedAttr(".nv"))
        vg.readIntAttr(".nv", &nv);
    vg.close();
    
    if(nv<4) {
        std::cout<<"\n cannot read velocities!";
        return nv;
    }
    
    createPoints(nv);
    return nv;
}

bool IVelocityFile::readFrameVelocity()
{
    useDocument();
    bool stat = true;
    HBase vg("/vel");
    const std::string sframe = boost::str( boost::format("%1%") % currentFrame() );
    if(vg.hasNamedData(sframe.c_str()))
        vg.readVector3Data(sframe.c_str(), numPoints(), velocities());
    else {
        std::cout<<" velocity file cannot read frame "<<currentFrame();
        stat = false;
    }
    vg.close();
    
    return stat;
}

BaseBuffer * IVelocityFile::velocityBuf() const
{ return m_vel; }
//;~
