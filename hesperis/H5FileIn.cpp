#include "H5FileIn.h"
#include <HTransform.h>
#include <HTriangleMeshGroup.h>
#include <ATriangleMesh.h>
#include <HFrameRange.h>
#include <boost/format.hpp>

H5FileIn::H5FileIn() : HFile() {}
H5FileIn::H5FileIn(const char * name) : HFile(name) {}

ATriangleMesh * H5FileIn::findBakedMesh(std::string & name)
{
    if(!entityExists("/.fr")) {
        std::cout<<" found no frame range\n";
        return 0;
    }
    
    HFrameRange fr("/.fr");
    fr.load(this);
    fr.close();
    
    HBase b("/");
    std::vector<std::string > gs;
    LsNames2<HBase, HTriangleMesh>(gs, &b);
    
    if(gs.size() < 1) {
        std::cout<<" found no mesh\n";
        return 0;
    }
    
    name = gs.back();
    HTriangleMesh m(name);
    
    if(!m.hasNamedChild(".geom")) {
        std::cout<<" found no geom\n";
        return 0;
    }
    
    const std::string geoName = m.childPath(".geom");
    HBase geo(geoName);
    if(!geo.hasNamedChild(".bake")) {
        std::cout<<" found no bake\n";
        return 0;
    }
    
    std::cout<<""<<geoName;
    
    m_bakeName = geo.childPath(".bake");
    HBase bak(m_bakeName);
    
    std::cout<<""<<m_bakeName;
    std::cout<<" n samples "<<bak.numChildren();
    
    std::vector<std::string > dataNames;
    bak.lsData(dataNames);
    if(dataNames.size() < 1) {
        std::cout<<" found no bake data\n";
        return 0;
    }
    
    //std::vector<std::string >::const_iterator it = dataNames.begin();
    //for(;it!=dataNames.end();++it) std::cout<<"\n"<<*it;
    
    ATriangleMesh * tri = new ATriangleMesh;
    m.load(tri);
    m.close();
    
    return tri;
}

void H5FileIn::frameBegin()
{ m_currentFrame = FirstFrame; }

bool H5FileIn::frameEnd() const
{ return m_currentFrame == LastFrame; }

void H5FileIn::nextFrame()
{ m_currentFrame++; }

bool H5FileIn::readFrame(Vector3F * dst, unsigned nv)
{
    useDocument();
    const std::string name = boost::str(boost::format("%1%/%2%") % m_bakeName % m_currentFrame);
    if(!entityExists(name)) {
        std::cout<<" cannot read frame "<<name<<"\n";
        return false;
    }
    
    
    return true;
}
//:~
