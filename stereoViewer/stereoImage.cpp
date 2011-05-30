#include "StereoImage.h"

#include <iostream>
#include <sstream>

using namespace std;

StereoImage::StereoImage():_data(0),_left_color(0),_right_color(0),_img_w(32),_img_h(32),
_has_left(0), _has_right(0)
{
}

void StereoImage::setSize(int w, int h)
{
    if(_img_w != w || _img_h != h)
    {
	_img_w = w;
	_img_h = h;
	if(_left_color) delete[] _left_color;
	if(_right_color) delete[] _right_color;
	if(_data) delete[] _data;
	_left_color = new float[_img_w*_img_h*4];
	_right_color = new float[_img_w*_img_h*4];
	_data = new char[_img_w*_img_h*4];
	_has_left = _has_right = 0;
    }
}

void StereoImage::setLeftImage(void *d)
{
	unsigned char* bits = (unsigned char *)d;

	for(int j=0; j< _img_h; j++) {
		for(int i=0; i< _img_w; i++) {
			
			_left_color[(j*_img_w+i)*4] = *bits/255.f;//Blue
			bits++;
			_left_color[(j*_img_w+i)*4+1] = *bits/255.f;//Green
			bits++;
			_left_color[(j*_img_w+i)*4+2] = *bits/255.f;//Red
			bits++;
			_left_color[(j*_img_w+i)*4+3] = *bits/255.f;//Alpha
			bits++;
			
		}
	}
	_has_left = 1;
}

void StereoImage::setRightImage(void *d)
{
	unsigned char* bits = (unsigned char *)d;

	for(int j=0; j< _img_h; j++) {
		for(int i=0; i< _img_w; i++) {
			
			_right_color[(j*_img_w+i)*4] = *bits/255.f;//Blue
			bits++;
			_right_color[(j*_img_w+i)*4+1] = *bits/255.f;//Green
			bits++;
			_right_color[(j*_img_w+i)*4+2] = *bits/255.f;//Red
			bits++;
			_right_color[(j*_img_w+i)*4+3] = *bits/255.f;//Alpha
			bits++;
			
		}
	}
	_has_right = 1;
}

void *StereoImage::display()
{
	if(!_data) return 0;
	
	int ipx;
	for(int j=0; j< _img_h; j++) {
		for(int i=0; i< _img_w; i++) {
			ipx = j*_img_w+i;
			_data[ipx*4] = _data[ipx*4+1] = _data[ipx*4+2] = _data[ipx*4+3] = 0;
			
			if(_has_left)
			{
			    _data[ipx*4] += _left_color[ipx*4]*127;//Blue
			    _data[ipx*4+1] += _left_color[ipx*4+1]*55;//Green
			    _data[ipx*4+2] += _left_color[ipx*4+2]*200;//Red
			    //_data[ipx*4+3] += 0;//Alpha
			}
			
			if(_has_right)
			{
			    _data[ipx*4] += _right_color[ipx*4]*127;//Blue
			    _data[ipx*4+1] += _right_color[ipx*4+1]*200;//Green
			    _data[ipx*4+2] += _right_color[ipx*4+2]*55;//Red
			    //_data[ipx*4+3] += 0;//Alpha
			}
			
		}
	}
	return _data;
}

