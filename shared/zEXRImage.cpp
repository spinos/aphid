#include "zEXRImage.h"
#include <cmath>

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
	float p[_rank];
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

ZEXRImage::ZEXRImage() : _pixels(0) {}

ZEXRImage::ZEXRImage(const char* filename) : _pixels(0)
{
	load(filename);
}

ZEXRImage::~ZEXRImage(void)
{
	clear();
}

char ZEXRImage::load(const char* filename)
{
	if(!isAnOpenExrFile(filename)) {
		std::cout<<"ERROR: "<<filename<<" is not an openEXR image\n";
		return 0;
	}
	std::cout<<"loading "<<filename<<"\n";
	clear();
	try {
	Imf::InputFile file(filename);
	Imath::Box2i dw = file.header().dataWindow(); 
	m_imageWidth = dw.max.x - dw.min.x + 1;
	m_imageHeight = dw.max.y - dw.min.y + 1;
	
	m_channelRank = RGB;
	const Imf::ChannelList &channels = file.header().channels(); 
	const Imf::Channel *channelAPtr = channels.findChannel("A");
	if(channelAPtr) m_channelRank = RGBA;
	
	readPixels(file);
	setupMipmaps();
	}
	catch (const std::exception &exc) { 
		return 0; 
	}
	_valid = 1;
	verbose();
	return _valid;
}

void ZEXRImage::clear()
{
    if(_pixels) delete[] _pixels;
	_pixels = 0;
	std::vector<ZEXRSampler *>::iterator it;
	for(it = _mipmaps.begin(); it != _mipmaps.end(); ++it)
		delete *it;
		
	_mipmaps.clear();
	_valid = 0;
}

bool ZEXRImage::isAnOpenExrFile (const char fileName[])
{ 
	std::ifstream f (fileName, std::ios_base::binary); 
	if(!f.is_open()) return 0;
	char b[4]; 
	f.read (b, sizeof (b)); 
	return !!f && b[0] == 0x76 && b[1] == 0x2f && b[2] == 0x31 && b[3] == 0x01; 
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
	
	const int size = getWidth() * getHeight() * m_channelRank;
	const int stride = m_channelRank;
	
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
							   
	file.setFrameBuffer (frameBuffer); 
	file.readPixels (dw.min.y, dw.max.y); 
}

void ZEXRImage::allWhite()
{
	m_imageWidth = m_imageHeight = 1024;
	const int size = m_imageWidth * m_imageHeight * 3;
	m_channelRank = RGB;
	_pixels = new half[size];
	
	for(int i = 0; i < size; i++) _pixels[i] = 1.f;
	setupMipmaps();
	_valid = 1;
}

void ZEXRImage::allBlack()
{
	m_imageWidth = m_imageHeight = 1024;
	const int size = m_imageWidth * m_imageHeight * 3;
	m_channelRank = RGB;
	_pixels = new half[size];
	
	for(int i = 0; i < size; i++) _pixels[i] = 0.f;
	setupMipmaps();
	_valid = 1;
}

int ZEXRImage::getNumMipmaps() const
{
	return _numMipmaps;
}

void ZEXRImage::verbose() const
{
	std::cout<<" image size: ("<<m_imageWidth<<", "<<m_imageHeight<<")\n";
	if(m_channelRank == RGB)
		std::cout<<" image channels: RGB\n";
	else
		std::cout<<" image channels: RGBA\n";
	std::cout<<" mipmap count: "<<_numMipmaps<<"\n";
	if(isValid())
		std::cout<<" exr image is verified\n";
}

void ZEXRImage::applyMask(BaseImage * another)
{
	const float du = 1.f / (float)m_imageWidth;
	const float dv = 1.f / (float)m_imageHeight;
	for(int j = 0; j < m_imageHeight; j++)
	{
		for(int i = 0; i < m_imageWidth; i++)
		{
			float u = du * i;
			float v = dv * j;
			
			const float msk = another->sampleRed(u, v);
			const float ori = _pixels[pixelLoc(u, v)];
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
	
	float hi[count];
	
	_mipmaps[_numMipmaps - levelHigh]->sample(u , v, count, hi);
	if(levelHigh == levelLow) {
		for(int k = 0; k < count; k++) dst[k] = hi[k];
		return;
	}
	
	float lo[count];
	 _mipmaps[_numMipmaps - levelLow]->sample(u , v, count, lo);
	
	float alpha = level - (int)level;
	
	for(int k = 0; k < count; k++) dst[k] = lo[k] * (1.f - alpha) + hi[k] * alpha;
}

void ZEXRImage::sample(float u, float v, int count, float * dst) const
{
    if(!_pixels) return;
	int loc = pixelLoc(u, v, true);
	for(int i = 0; i < count; i++)
		dst[i] = _pixels[loc + i];
}
