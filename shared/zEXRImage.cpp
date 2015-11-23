#include "zEXRImage.h"
#include <ImfHeader.h>
#include <ImfRgbaFile.h>
#include <ImfInputFile.h>
#include <ImfOutputFile.h>
#include <ImfChannelList.h>
#include <ImfArray.h>

#include <ImfBoxAttribute.h>
#include <ImfChannelListAttribute.h>
#include <ImfCompressionAttribute.h>
#include <ImfChromaticitiesAttribute.h>
#include <ImfFloatAttribute.h>
#include <ImfEnvmapAttribute.h>
#include <ImfDoubleAttribute.h>
#include <ImfIntAttribute.h>
#include <ImfLineOrderAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfOpaqueAttribute.h>
#include <ImfStringAttribute.h>
#include <ImfVecAttribute.h>

double Log2( double n )  
{  
    // log(n)/log(2) is log2.  
    return log( n ) / log( 2.0 );  
}

ZEXRSampler::ZEXRSampler() : _pixels(0)
{
}

ZEXRSampler::~ZEXRSampler()
{
	if(_pixels) delete[] _pixels;
}

void ZEXRSampler::setPixels(ZEXRImage * src)
{
	_rank = src->m_channelRank;
	if(_rank>4) _rank = 4;
	float * p = new float[_rank];
	_pixels = new half[_width * _width * _rank];
	float u, v;
	const float d = 1.f / (float)_width;
	const int stride = _width * _rank;
	int i, j, k;
	for(j =0; j < _width; j++) {
		v = d * j;
		for(i =0; i < _width; i++) {
			u = d * i;
			src->sample(u, v, _rank, p);
			for(k = 0; k < _rank; k++)
				_pixels[stride * j + i * _rank + k] = p[k];
		}
	}
	delete[] p;
}

void ZEXRSampler::setWidth(int w)
{
	_width = 2;
	while(_width * 2 <= w) 
		_width *= 2;
}

half* ZEXRSampler::getPixels() const
{
	return _pixels;
}

int ZEXRSampler::getWidth() const
{
	return _width;
}

void ZEXRSampler::reduceFrom(ZEXRSampler * src)
{
	_rank = src->_rank;
	_width =  src->_width / 2;
	_pixels = new half[_width * _width * _rank];
	int i, j, k, loc;
	const int w = src->_width;
	half * p = src->_pixels;
	for(j=0; j < _width; j++) {
		for(i=0; i < _width; i++) {
			loc = j * _width * _rank + i * _rank;
			for(k = 0; k < _rank; k++)
				_pixels[loc + k] = p[j*2 *      w * _rank + i*2 * _rank + k] * 0.25f
								+ p[j*2 *       w * _rank + (i*2 + 1) * _rank + k] * 0.25f
								+ p[(j*2 + 1) * w * _rank + (i*2 + 1) * _rank + k] * 0.25f
								+ p[(j*2 + 1) * w * _rank + i*2 * _rank + k] * 0.25f;
		}
	}
}

void ZEXRSampler::sample(float u, float v, int count, float * dst) const
{
	float gridu = (_width - 1) * u;
	float gridv = (_width - 1) * v;
	
	int x0 = gridu;
	int x1 = x0 + 1;
	if(x1 > _width - 1) x1 = _width - 1;
	int y0 = gridv;
	int y1 = y0 + 1;
	if(y1 > _width - 1) y1 = _width - 1;
	
	float alphau = gridu - x0;
	float alphav = gridv - y0;
	
	const int ystride = _width * _rank;
	for(int k = 0; k < count; k++) 
		dst[k] = (float)((_pixels[y0 * ystride + x0 * _rank + k] * ( 1.f - alphau) + _pixels[y0 * ystride + x1 * _rank + k] * alphau) * (1.f - alphav)
					+ (_pixels[y1 * ystride + x0 * _rank + k] * ( 1.f - alphau) + _pixels[y1 * ystride + x1 * _rank + k] * alphau) * alphav);
}

ZEXRImage::ZEXRImage() : _pixels(0), m_zData(0), m_hasMipmap(false) {}

ZEXRImage::ZEXRImage(const char * filename) : m_hasMipmap(false), BaseImage(filename) {}

ZEXRImage::ZEXRImage(bool loadMipmap) : _pixels(0), m_zData(0), m_hasMipmap(loadMipmap) {}

ZEXRImage::~ZEXRImage()
{
}

bool ZEXRImage::doRead(const std::string & filename)
{
	if(!isAnOpenExrFile(filename)) {
		std::cout<<"ERROR: "<<filename<<" is not an openEXR image\n";
		return false;
	}
	
	try {
	Imf::InputFile file(filename.c_str());
	Imath::Box2i dw = file.header().dataWindow(); 
	m_imageWidth = dw.max.x - dw.min.x + 1;
	m_imageHeight = dw.max.y - dw.min.y + 1;
	
	const Imf::ChannelList &channels = file.header().channels(); 
	
	const Imf::Channel *rChannel = channels.findChannel("R");
	if(rChannel) m_channelRank = RED;
	const Imf::Channel *gChannel = channels.findChannel("G");
	const Imf::Channel *bChannel = channels.findChannel("B");
	if(rChannel && gChannel && bChannel) m_channelRank = RGB;
	const Imf::Channel *aChannel = channels.findChannel("A");
	if(rChannel && gChannel && bChannel && aChannel) m_channelRank = RGBA;
	
	const Imf::Channel *channelZPtr = channels.findChannel("Z");
	if(channelZPtr) {
	    m_channelRank = RGBAZ;
	    m_zChannelName = "Z";
	}
	else {
	    if(findZChannel(file))
	        m_channelRank = RGBAZ;
	}
	
		readPixels(file);
		if(m_hasMipmap) setupMipmaps();
	}
	catch (const std::exception &exc) { 
		std::cout<<"ERROR: "<<filename<<" cannot be loaded as an openEXR image\n";
		return false; 
	}

	return true;
}

bool ZEXRImage::findZChannel(Imf::InputFile & file)
{
	m_zChannelName = "";
	const Imf::ChannelList &channels = file.header().channels(); 
	Imf::ChannelList::ConstIterator it = channels.begin();
	for(; it!= channels.end(); ++it) {
	    if(std::string(it.name()).find_last_of('Z') != std::string::npos) {
	        m_zChannelName = it.name();
	        break;
	    }
	}
	
	if(m_zChannelName.size() < 1) return false;
	
	const Imf::Channel *channelZPtr = channels.findChannel(m_zChannelName.c_str());
/// http://code.woboq.org/appleseed/appleseed/openexr/include/OpenEXR/ImfPixelType.h.html#Imf::PixelType
	if(channelZPtr->type == Imf::FLOAT)
	    return true;

	return false;
}

void ZEXRImage::doClear()
{
    if(_pixels) delete[] _pixels;
	_pixels = 0;
	if(m_zData) delete[] m_zData;
	m_zData = 0;
	std::vector<ZEXRSampler *>::iterator it;
	for(it = _mipmaps.begin(); it != _mipmaps.end(); ++it)
		delete *it;
		
	_mipmaps.clear();
	BaseImage::doClear();
}

const char * ZEXRImage::formatName() const
{
	return "OpenEXR";
}

bool ZEXRImage::isAnOpenExrFile (const std::string& fileName)
{ 
	std::ifstream f (fileName.c_str(), std::ios_base::binary); 
	if(!f.is_open()) return 0;
	char b[4]; 
	f.read (b, sizeof (b)); 
	f.close();
	return !!f && b[0] == 0x76 && b[1] == 0x2f && b[2] == 0x31 && b[3] == 0x01; 
}

void ZEXRImage::listExrChannelNames(const std::string& filename, std::vector<std::string>& dst)
{
    if(!isAnOpenExrFile(filename)) {
		std::cout<<"ERROR: "<<filename<<" is not an openEXR image\n";
		return;
	}
	try {
	    //std::cout<<"begin exr channels:\n";
	    dst.clear();
	Imf::InputFile file(filename.c_str());
	const Imf::ChannelList &channels = file.header().channels(); 
	Imf::ChannelList::ConstIterator it = channels.begin();
	for(; it!= channels.end(); ++it) {
	    //std::cout<<"channel name "<<it.name()<<"\n";
	    dst.push_back(it.name());
	}
	    //std::cout<<"end exr channels:\n";
	}
	catch (const std::exception &exc) { 
		std::cout<<"ERROR: "<<filename<<" cannot be loaded as an openEXR image\n";
		return; 
	}
}

void ZEXRImage::setupMipmaps()
{
	_numMipmaps = Log2((double)m_imageWidth);
	
	ZEXRSampler* mipmap = new ZEXRSampler();
	mipmap->setWidth(m_imageWidth);
	mipmap->setPixels(this);
	_mipmaps.push_back(mipmap);

	for(int i=1; i <_numMipmaps; i++) {
		ZEXRSampler* submap = new ZEXRSampler();
		submap->reduceFrom(_mipmaps[i-1]);
		_mipmaps.push_back(submap);
		//printf("mipmap %i\n", _mipmaps[i]->getWidth());
	}
}

void ZEXRImage::readPixels(Imf::InputFile& file)
{
	Imath::Box2i dw = file.header().dataWindow();
	
	int colorRank = m_channelRank;
	if(colorRank > 4) colorRank = 4;
	const int size = getWidth() * getHeight() * colorRank;
	const int stride = colorRank;
	
	_pixels = new half[size];
	
	Imf::FrameBuffer frameBuffer; 
	frameBuffer.insert ("R",                                  // name 
		Imf::Slice (Imf::HALF,                          // type 
							   (char *) _pixels, 
							   sizeof (*_pixels) * stride,    // xStride 
							   sizeof (*_pixels) * getWidth() * stride));                         // fillValue 
	_pixels++;
	frameBuffer.insert ("G",                                  // name 
		Imf::Slice (Imf::HALF,                          // type 
							   (char *) _pixels, 
							   sizeof (*_pixels) * stride,    // xStride 
							   sizeof (*_pixels) * getWidth() * stride));
	_pixels++;
	frameBuffer.insert ("B",                                  // name 
		Imf::Slice (Imf::HALF,                          // type 
							   (char *) _pixels, 
							   sizeof (*_pixels) * stride,    // xStride 
							   sizeof (*_pixels) * getWidth() * stride));
							   
	if(stride > 3) {
		_pixels++;
		frameBuffer.insert ("A",                                  // name 
		Imf::Slice (Imf::HALF,                          // type 
							   (char *) _pixels, 
							   sizeof (*_pixels) * stride,    // xStride 
							   sizeof (*_pixels) * getWidth() * stride));
		_pixels--;
	}
	_pixels--;
	_pixels--;
	
	if(m_channelRank == RGBAZ) {
	    m_zData = new float[getWidth() * getHeight()];
		frameBuffer.insert (m_zChannelName.c_str(),                                  // name 
		Imf::Slice (Imf::FLOAT,                          // type 
							   (char *) m_zData, 
							   sizeof (*m_zData) * 1,    // xStride 
							   sizeof (*m_zData) * getWidth() * 1));                         // fillValue 
							   
	}
							   
	file.setFrameBuffer (frameBuffer); 
	file.readPixels (dw.min.y, dw.max.y);
}

void ZEXRImage::readZ(Imf::InputFile& file)
{
    const Imf::ChannelList &channels = file.header().channels(); 
	const Imf::Channel *channelZPtr = channels.findChannel(m_zChannelName.c_str());
// http://code.woboq.org/appleseed/appleseed/openexr/include/OpenEXR/ImfPixelType.h.html#Imf::PixelType
	if(channelZPtr->type != Imf::FLOAT) {
	    std::cout<<"Warning: z channel is not float\n";
	    // std::cout<<"z type "<<channelZPtr->type<<"\n";
	    return;
	}
	
	Imath::Box2i dw = file.header().dataWindow();
	
	const int size = getWidth() * getHeight();
	const int stride = 1;
	
	m_zData = new float[size];
	
	Imf::FrameBuffer frameBuffer; 
	frameBuffer.insert (m_zChannelName.c_str(),                                  // name 
		Imf::Slice (Imf::FLOAT,                          // type 
							   (char *) m_zData, 
							   sizeof (*m_zData) * stride,    // xStride 
							   sizeof (*m_zData) * getWidth() * stride));                         // fillValue 
							   
	file.setFrameBuffer (frameBuffer); 
	file.readPixels (dw.min.y, dw.max.y);
	
	//for(int i=0; i < size; i++) {
	//    if(m_zData[i] < 10000) std::cout<<" "<<m_zData[i];
	//}
}

void ZEXRImage::allWhite()
{
	m_imageWidth = m_imageHeight = 1024;
	const int size = m_imageWidth * m_imageHeight * 3;
	m_channelRank = RGB;
	_pixels = new half[size];
	
	for(int i = 0; i < size; i++) _pixels[i] = 1.f;
	setupMipmaps();
	setOpened();
}

void ZEXRImage::allBlack()
{
	m_imageWidth = m_imageHeight = 1024;
	const int size = m_imageWidth * m_imageHeight * 3;
	m_channelRank = RGB;
	_pixels = new half[size];
	
	for(int i = 0; i < size; i++) _pixels[i] = 0.f;
	setupMipmaps();
	setOpened();
}

int ZEXRImage::getNumMipmaps() const
{
	return _numMipmaps;
}

void ZEXRImage::applyMask(BaseImage * another)
{
	const float du = 1.f / (float)m_imageWidth;
	const float dv = 1.f / (float)m_imageHeight;
	int pixelRank = m_channelRank;
	if(pixelRank > 4) pixelRank = 4;
	for(int j = 0; j < m_imageHeight; j++)
	{
		for(int i = 0; i < m_imageWidth; i++)
		{
			float u = du * i;
			float v = dv * j;
			
			const float msk = another->sampleRed(u, v);
			const float ori = _pixels[pixelLoc(u, v, 0, pixelRank)];
			setRed(u, v, ori * msk);
		}
	}
}

void ZEXRImage::sample(float u, float v, float level, int count, float * dst) const
{
	int levelHigh = (int)level + 1;
	int levelLow =  (int)level;
	if(levelHigh > _numMipmaps) levelHigh = _numMipmaps;
	if(levelHigh < 1) levelHigh = 1;
	
	if(levelLow > _numMipmaps) levelLow = _numMipmaps;
	if(levelLow < 1) levelLow = 1;
	
	float * hi = new float[count];
	
	_mipmaps[_numMipmaps - levelHigh]->sample(u , v, count, hi);
	if(levelHigh == levelLow) {
		for(int k = 0; k < count; k++) dst[k] = hi[k];
		delete[] hi;
		return;
	}
	
	float * lo = new float[count];
	 _mipmaps[_numMipmaps - levelLow]->sample(u , v, count, lo);
	
	float alpha = level - (int)level;
	
	for(int k = 0; k < count; k++) dst[k] = lo[k] * (1.f - alpha) + hi[k] * alpha;
	delete[] hi;
	delete[] lo;
}

void ZEXRImage::sample(float u, float v, int count, float * dst) const
{
    if(!_pixels) return;
	int loc = pixelLoc(u, v, true, count);
	for(int i = 0; i < count; i++)
		dst[i] = _pixels[loc + i];
}

bool ZEXRImage::getTile(float * dst, int ind, int tileSize, int rank) const
{
	const int dimx = m_imageWidth / tileSize;
	const int dimy = m_imageHeight / tileSize;
	int rind = ind % (dimx * dimy);
	int y = rind / dimx;
	int x = rind - y * dimx;
	return getTile(dst, x, y, tileSize, rank);
}

bool ZEXRImage::getTile(float * dst, int x, int y, int tileSize, int rank) const
{	
	int colorRank = m_channelRank;
	if(colorRank > 4) colorRank = 4;
	
	half *line = &_pixels[(y * tileSize * m_imageWidth + x * tileSize) * colorRank];
	int i, j, k;
	for(j=0;j<tileSize; j++) {
		for(i=0;i<tileSize; i++) {
			for(k=0;k<rank;k++) {
				dst[(j * tileSize + i) * rank + k] = line[i * colorRank + k];
				
			}
			
			// dst[(j * tileSize + i) * rank + 0] = 1.f / tileSize * i;

		}
		line += m_imageWidth * colorRank;
	}
	
	return true;
}
