#ifndef STEREOIMAGE_H
#define STEREOIMAGE_H

// Define the interface to the stereo image library.

class StereoImage {
	

public:
    StereoImage();
    
    void setSize(int w, int h);
    void setLeftImage(void *d);
    void setRightImage(void *d);

    void *display();
    
    
private:
    char *_data;
    float* _left_color;
    float* _right_color;
    int _img_w, _img_h;
    char _has_left, _has_right;
};
#endif        //  #ifndef STEREOIMAGE_H

