#include "APlaybackFile.h"
#include <HFrameRange.h>
#include <boost/format.hpp>

APlaybackFile::APlaybackFile() : HFile() 
{}

APlaybackFile::APlaybackFile(const char * name) : HFile(name) 
{}

APlaybackFile::~APlaybackFile() {}

bool APlaybackFile::writeFrameRange(AFrameRange * src)
{
    if(!src->isValid()) return false;
    useDocument();
	HFrameRange g("/.fr");
	g.save(src);
	g.close();
    return true;
}

bool APlaybackFile::readFrameRange()
{
    useDocument();
    if(!entityExists("/.fr")) {
        std::cout<<"\n error: playback file has no frame range\n";
        AFrameRange::reset();
        return false;
    }
    
    HFrameRange fr("/.fr");
    fr.load(this);
    fr.close();

    return true;
}

void APlaybackFile::verbose() const
{
    std::cout<<"\n playback range: ("<<this->FirstFrame
    <<","<<this->LastFrame
    <<")";
}
//:~
