#include "ExrImage.h"
#include <OpenEXR/half.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfNamespace.h>
#include <OpenEXR/ImathNamespace.h>

namespace IMF = OPENEXR_IMF_NAMESPACE;
using namespace IMF;
using namespace IMATH_NAMESPACE;

namespace aphid {
BaseImage::ChannelRank checkExrColors(const Header & head)
{
    const ChannelList &channels = head.channels(); 
	
	const Channel *rChannel = channels.findChannel("R");
	if(!rChannel) return BaseImage::None;
	if(rChannel->type != HALF) return BaseImage::None;
    
    const Channel *gChannel = channels.findChannel("G");
    if(!gChannel) return BaseImage::RED;
	if(gChannel->type != HALF) return BaseImage::RED;
    
	const Channel *bChannel = channels.findChannel("B");
	if(!bChannel) return BaseImage::RED;
	if(bChannel->type != HALF) return BaseImage::RED;
	
    const Channel *aChannel = channels.findChannel("A");
	if(!aChannel) return BaseImage::RGB;
	if(aChannel->type != HALF) return BaseImage::RGB;
	return BaseImage::RGBA;
}

void readColor(half * data, InputFile& file, const Box2i & dw, int stride)
{
    const int width  = dw.max.x - dw.min.x + 1;
    
	FrameBuffer frameBuffer; 
	frameBuffer.insert ("R",                                  // name 
		Slice (HALF,                          // type 
							   (char *) data, 
							   sizeof (half) * stride,    // xStride 
							   sizeof (half) * width * stride));                         // fillValue 
	data++;
	frameBuffer.insert ("G",                                  // name 
		Slice (HALF,                          // type 
							   (char *) data, 
							   sizeof (half) * stride,    // xStride 
							   sizeof (half) * width * stride));
	data++;
	frameBuffer.insert ("B",                                  // name 
		Slice (HALF,                          // type 
							   (char *) data, 
							   sizeof (half) * stride,    // xStride 
							   sizeof (half) * width * stride));
							   
	if(stride > 3) {
		data++;
		frameBuffer.insert ("A",                                  // name 
		Imf::Slice (HALF,                          // type 
							   (char *) data, 
							   sizeof (half) * stride,    // xStride 
							   sizeof (half) * width * stride));
	}
	
							   
	file.setFrameBuffer (frameBuffer); 
	file.readPixels (dw.min.y, dw.max.y);
}


ExrImage::ExrImage() : _pixels(0) {}

ExrImage::~ExrImage() 
{ clear(); }

void ExrImage::clear()
{ 
	if(_pixels) {
		delete[] _pixels;
	}
}

bool ExrImage::readImage(const std::string & filename)
{
	clear();
	
    if(!IsOpenExrFile(filename)) {
		std::cout<<"\n ExrImage ERROR: "<<filename<<" is not an openEXR image\n";
		return false;
	}
	
	InputFile infile(filename.c_str());
	
	const Header & inhead = infile.header();
	ChannelRank rnk = checkExrColors(inhead);
	if(rnk == None) {
		std::cout<<"\n ExrImage ERROR: "<<filename<<" has no R channel\n";
		return false;
	}
	
	setChannelRank(rnk);
	
	const Box2i &dw = infile.header().dataWindow(); 
	setWidth(dw.max.x - dw.min.x + 1);
	setHeight(dw.max.y - dw.min.y + 1);
	
	int colorRank = channelRank();
	if(colorRank > 4) colorRank = 4;
	const int size = getWidth() * getHeight() * colorRank;
	const int stride = colorRank;
	
	_pixels = (char *)malloc(size * sizeof(half));
	readColor((half *)_pixels, infile, dw, colorRank);
	return true;
}

/// read R,G,B separatedly

bool ExrImage::getTile(float * dst, const int ind, const int tileSize, int rank) const
{
    int x, y;
    tileCoord(ind, tileSize, x, y);
	const int w = getWidth();
	int colorRank = channelRank();
	if(colorRank > 4) colorRank = 4;
	
	const int stride = tileSize * tileSize;
	
	half * hp = (half *)_pixels;
	
	half *line = &hp[(y * tileSize * w + x * tileSize) * colorRank];
	int i, j, k;
	for(j=0;j<tileSize; j++) {
		for(i=0;i<tileSize; i++) {
			for(k=0;k<rank;k++) {
				dst[(j * tileSize + i) + stride * k] = line[i * colorRank + k];
				
			}
		}
		line += w * colorRank;
	}
	
	return true;
}

void ExrImage::tileCoord(const int ind, const int tileSize, int & x, int & y) const
{
    const int dimx = getWidth() / tileSize;
	const int dimy = getHeight() / tileSize;
	int rind = ind % (dimx * dimy);
    
	y = rind / dimx;
	x = rind - y * dimx;
}

BaseImage::IFormat ExrImage::formatName() const
{ return FExr; }
	
bool ExrImage::IsOpenExrFile(const std::string& filename)
{
    std::ifstream f (filename.c_str(), std::ios_base::binary); 
	if(!f.is_open()) 
		return false;
	
	char b[4]; 
	f.read (b, sizeof (b)); 
	return !!f && b[0] == 0x76 && b[1] == 0x2f && b[2] == 0x31 && b[3] == 0x01; 
}

void ExrImage::sample(float u, float v, int count, float * dst) const
{
	const int w = getWidth();
	int colorRank = channelRank();
	if(colorRank > 4) colorRank = 4;
	
	int lx = u * w;
	if(lx < 0) {
		lx = 0;
	}
	if(lx > w-1 ) {
		lx = w-1;
	}
/// flip vertically, top-left is origin
	int ly = (1.f - v) * getHeight();
	if(ly < 0) {
		ly = 0;
	}
	if(ly > getHeight()-1 ) {
		ly = getHeight()-1;
	}
	
	half * hp = (half *)_pixels;
	half *line = &hp[(ly * w + lx) * colorRank];
	for(int i=0;i<count;++i) {
		dst[i] = line[i];
	}
	
}

}