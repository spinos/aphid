#ifndef SCAM_IMPLEMENT_H
#define SCAM_IMPLEMENT_H

extern "C" void initTexture(int width, int height, unsigned char*pImage, unsigned char*outImage);
extern "C" void compactImage(
    int width, 
    int height, 
    unsigned char *pImage, 
    unsigned char *outImage, 
    char needSort = 0);

extern "C" uint scanExclusive(
    uint *d_Dst,
    uint *d_Src,
    uint batchSize,
    uint arrayLength
);

extern "C" void checkScanResult(
    uint *d_scanResult, 
    uint *d_count, 
    uint numElement
);

#endif        //  #ifndef SCAM_IMPLEMENT_H

