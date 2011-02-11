#include "FooImage.h"

#include <iostream>
#include <sstream>

using namespace std;

FooImage::FooImage(const char *w):_red(1.f),_data(0),_img_w(32),_img_h(32),
_color(0)
{
	_the_word = w;
	
}

void FooImage::setSize(int w, int h)
{
	_img_w = w;
	_img_h = h;
	if(_data) delete[] _data;
	_data = new char[_img_w*_img_h*4];
	if(_color) delete[] _color;
	_color = new float[_img_w*_img_h*4];
}

void FooImage::addColor(void *d)
{
	unsigned char* bits = (unsigned char *)d;

	for(int j=0; j< _img_h; j++) {
		for(int i=0; i< _img_w; i++) {
			
			_color[(j*_img_w+i)*4] = *bits/255.f;//Blue
			bits++;
			_color[(j*_img_w+i)*4+1] = *bits/255.f;//Green
			bits++;
			_color[(j*_img_w+i)*4+2] = *bits/255.f;//Red
			bits++;
			_color[(j*_img_w+i)*4+3] = *bits/255.f;//Alpha
			bits++;
			
		}
	}
}

void FooImage::setRed(float red)
{
	_red = red;
}

void *FooImage::display()
{
	if(!_data) return 0;
	
	int ipx;
	for(int j=0; j< _img_h; j++) {
		for(int i=0; i< _img_w; i++) {
			ipx = j*_img_w+i;
			_data[ipx*4] = _color[ipx*4]*255;//Blue
			_data[ipx*4+1] = _color[ipx*4+1]*255;//Green
			_data[ipx*4+2] = _color[ipx*4+2]*_red*255;//Red
			_data[ipx*4+3] = 0;//Alpha
			
		}
	}
	return _data;
}

