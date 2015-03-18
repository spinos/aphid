#ifndef STRIPECOMPRESSEDIMAGE_H
#define STRIPECOMPRESSEDIMAGE_H

#include <map>

#define STRIPE_SIZE 256
#define STRIPE_RGBA_BYTE_SIZE 2048 // 256 * 4 * 2

struct StripeRGBAZ {
    char rgba[STRIPE_RGBA_BYTE_SIZE]; // 256 * 4 * 2
    float z[STRIPE_SIZE];
};

class ZEXRImage;

class StripeCompressedRGBAZImage 
{
public:
    StripeCompressedRGBAZImage();
    virtual ~StripeCompressedRGBAZImage();
    
    void compress(ZEXRImage * img);
    void decompress(char * rgbaBuf, char * zBuf, unsigned numPix) const;
    const unsigned numPixels() const;
    const unsigned width() const;
    const unsigned height() const;
private:
    bool isStripeOccupied(float * depth, unsigned n);
    unsigned getStripeCount(unsigned numPix) const;
    int getNumPixInStripe(unsigned i, unsigned count, unsigned numPix) const;
private:
    std::map<unsigned, StripeRGBAZ * > m_stripes;
    unsigned m_width, m_height;
};
#endif        //  #ifndef STRIPECOMPRESSEDIMAGE_H
