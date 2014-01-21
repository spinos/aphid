#include "AnorldFunc.h"
#include <iostream>
#include <BaseCamera.h>
AnorldFunc::AnorldFunc() 
{
}

AnorldFunc::~AnorldFunc() 
{
}

void AnorldFunc::render()
{
    RenderEngine::render();
}

void AnorldFunc::logArnoldVersion() const
{
#ifdef WIN32
    std::clog<<"\nRenderer is "<<AiGetVersionInfo()<<"\n";
#endif  
}

void AnorldFunc::loadPlugin(const char * fileName)
{
#ifdef WIN32
    AiLoadPlugins(fileName);
#endif
}
#ifdef WIN32
void AnorldFunc::setMatrix(const Matrix44F & src, AtMatrix & dst) const
{
    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++)
            dst[i][j] = src.M(i, j);
}

void AnorldFunc::logAStrArray(AtArray *arr)
{
	std::cout<<"n elm "<<arr->nelements<<"\n";
    for(unsigned i = 0; i < arr->nelements; i++)
        std::cout<<" "<<AiArrayGetStr(arr, i);
}

void AnorldFunc::logAMatrix(AtMatrix matrix)
{
	for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++)
            std::cout<<"m["<<i<<"]["<<j<<"] = "<<matrix[i][j]<<"\n";
}

void AnorldFunc::logRenderError(int status)
{
	if (status == AI_SUCCESS) {
		std::cout<<"rendered without error\n";
		return;
	}
	switch (status) {
		case AI_ABORT:
			std::cout<<"render aborted";
			break;
		case AI_ERROR_WRONG_OUTPUT:
			std::cout<<"can't open output file";
			break;
		case AI_ERROR_NO_CAMERA:
			std::cout<<"camera not definede";
			break;
		case AI_ERROR_BAD_CAMERA:
			std::cout<<"bad camera datae";
			break;
		case AI_ERROR_VALIDATION:
			std::cout<<"usage not validatede";
			break;
		case AI_ERROR_RENDER_REGION:
			std::cout<<"invalid render regione";
			break;
		case AI_ERROR_OUTPUT_EXISTS:
			std::cout<<"output file already existse";
			 break;
		case AI_ERROR_OPENING_FILE:
			std::cout<<"can't open filee";
			 break;
		case AI_INTERRUPT:
			std::cout<<"render interrupted by usere";
			 break;
		case AI_ERROR_UNRENDERABLE_SCENEGRAPH:
			std::cout<<"unrenderable scenegraphe";
			 break;
		case AI_ERROR_NO_OUTPUTS:
			std::cout<<"no rendering outputs";
			 break;
		default:
			std::cout<<"Unspecified Ai Error";
			break;
	}
}
#endif