#include "APlaybackFile.h"
#include <HFrameRange.h>
#include <boost/format.hpp>

APlaybackFile::APlaybackFile() : HFile() 
{ m_numFramesPlayed = 0; }

APlaybackFile::APlaybackFile(const char * name) : HFile(name) 
{ m_numFramesPlayed = 0; }

APlaybackFile::~APlaybackFile() {}

void APlaybackFile::frameBegin()
{ m_currentFrame = FirstFrame; }

bool APlaybackFile::isFrameEnd() const
{ return m_currentFrame == LastFrame; }

void APlaybackFile::nextFrame()
{ m_currentFrame++; }

const int APlaybackFile::currentFrame() const
{ return m_currentFrame; }

bool APlaybackFile::isFrameBegin() const
{ return m_currentFrame == FirstFrame; }

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
        std::cout<<" playback file has no frame range\n";
        AFrameRange::reset();
        return false;
    }
    
    HFrameRange fr("/.fr");
    fr.load(this);
    fr.close();
    
    return true;
}

void APlaybackFile::beginCountNumFramesPlayed()
{ m_numFramesPlayed = 0; }

const int APlaybackFile::numFramesPlayed() const
{ return m_numFramesPlayed; }

const bool APlaybackFile::allFramesPlayed() const
{ return numFramesPlayed() >= numFramesInRange(); }

void APlaybackFile::countNumFramesPlayed()
{ m_numFramesPlayed++; }
//:~
