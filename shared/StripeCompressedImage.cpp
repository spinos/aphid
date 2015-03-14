#include "StripeCompressedImage.h"

static float EmptyZ[STRIPE_SIZE];

StripeCompressedRGBAZImage::StripeCompressedRGBAZImage() {}
StripeCompressedRGBAZImage::~StripeCompressedRGBAZImage() 
{
    std::map<short, StripeRGBAZ * >::iterator it = m_stripes.begin();
    for(;it != m_stripes.end(); ++it)
        delete it->second;
    m_stripes.clear();
}

void StripeCompressedRGBAZImage::compress(ZEXRImage * img)
{
    const unsigned numPix = img->getWidth() * img->getHeight();
    const unsigned stripeCount = getStripeCount(numPix);
    
    std::cout<<" image has "<<stripeCount<<" stripes\n";
    
    float * depth = img->m_zData;
    char * rgba = (char *)img->_pixels;
    
    unsigned i, npixToCp;
    for(i=0; i<stripeCount; i++) {
        if(isStripeOccupied(depth)) {
            StripeRGBAZ * s = new StripeRGBAZ;
            
            npixToCp = getNumPixInStripe(i, stripeCount, numPix);
            
            memcpy(s.z, depth, npixToCp * 4);
            memcpy(s.rgba, rgba, npixToCp * 8);
            
            m_stripes[i] = s;
        }
        rgba += STRIPE_RGBA_BYTE_SIZE;
        depth += STRIPE_SIZE;
    }
    
    std::cout<<" read "<<m_stripes.size()<<" stripes\n";
}

bool StripeCompressedRGBAZImage::isStripeOccupied(float * depth)
{
    int i;
    for(i=0; i< STRIPE_SIZE; i++) {
        if(depth[i] > 0.1f && depth[i] < 1e8) return true; 
    }
    return false;
}

void StripeCompressedRGBAZImage::decompress(char * rgbaBuf, char * zBuf, unsigned numPix) const
{
    unsigned i, npixToCp;
    const unsigned stripeCount = getStripeCount(numPix);
    
    std::cout<<" image has "<<stripeCount<<" stripes\n";
    
    for(i = 0; i < stripeCount; i++) {
        npixToCp = getNumPixInStripe(i, stripeCount, numPix);
        memcpy(&zBuf[i * STRIPE_SIZE * 4], EmptyZ, npixToCp * 4);
    }
    
    std::map<short, StripeRGBAZ * >::const_iterator it = m_stripes.begin();
    for(;it != m_stripes.end(); ++it) {
        i = it->first;
        StripeRGBAZ * s = it->second;
        
        npixToCp = getNumPixInStripe(i, stripeCount, numPix);
        
        memcpy(&zBuf[i * STRIPE_SIZE * 4], s.z, npixToCp * 4);
        memcpy(&rgbaBuf[i * STRIPE_RGBA_BYTE_SIZE], s.rgba, npixToCp * 8);    
    }
    
    std::cout<<" write "<<m_stripes.size()<<" stripes\n";
}

unsigned StripeCompressedRGBAZImage::getStripeCount(unsigned numPix) const
{ return (numPix>>8) + ((numPix & 255) != 0); }

int StripeCompressedRGBAZImage::getNumPixInStripe(unsigned i, unsigned count, unsigned numPix) const
{
    if(i < count - 1) return STRIPE_SIZE;
    if((numPix & 255) != 0)
           return numPix & 255;
    return STRIPE_SIZE;
}

