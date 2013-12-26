#include <ai.h>

#include <iostream>
#include <vector>

void logAStrArray(AtArray *arr)
{
	std::cout<<"n elm "<<arr->nelements<<"\n";
    for(unsigned i = 0; i < arr->nelements; i++)
        std::cout<<" "<<AiArrayGetStr(arr, i);
}

void logAMatrix(AtMatrix matrix)
{
	for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++)
        std::cout<<"m["<<i<<"]["<<j<<"] = "<<matrix[i][j]<<"\n";
}

void logRenderError(int error)
{
	if (ai_status == AI_SUCCESS) {
		std::cout<<"no error\n";
		return;
	}
	switch (ai_status) {
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

int main(int argc, char *argv[])
{
    AiBegin();
    
    AtNode* options = AiNode("options");
    AtArray* outputs  = AiArrayAllocate(1, 1, AI_TYPE_STRING);
    AiArraySetStr(outputs, 0, "RGBA RGBA output:gaussian_filter output:exr");
    AiNodeSetArray(options, "outputs", outputs);
    
    logAStrArray(AiNodeGetArray(options, "outputs"));
    
    AiNodeSetInt(options, "xres", 320);
    AiNodeSetInt(options, "yres", 240);
    AiNodeSetInt(options, "AA_samples", 3);
	
    AtNode* driver = AiNode("driver_exr");
    AiNodeSetStr(driver, "name", "output:exr");
    AiNodeSetStr(driver, "filename", "output.exr");
    AiNodeSetFlt(driver, "gamma", 2.2f);
    
    AtNode * camera = AiNode("persp_camera");
    AiNodeSetStr(camera, "name", "/obj/cam");
    AiNodeSetFlt(camera, "fov", 54.f);
    AiNodeSetFlt(camera, "near_clip", 0.100000001);
    AiNodeSetFlt(camera, "far_clip", 10000);
    AtMatrix matrix;
    AiM4Identity(matrix);
    matrix[3][2] = 10.f;
	AiNodeSetMatrix(camera, "matrix", matrix);
	
    AiNodeGetMatrix(camera, "matrix", matrix);
    logAMatrix(matrix);
	
	AiNodeSetPtr(options, "camera", camera);
    
	AtNode * filter = AiNode("gaussian_filter");
    AiNodeSetStr(filter, "name", "output:gaussian_filter");

    AtNode * standard = AiNode("standard");
    AiNodeSetStr(standard, "name", "/shop/standard1");

    AtNode * sphere = AiNode("sphere");
    AiNodeSetPtr(sphere, "shader", standard);
    
    AtNode * light = AiNode("point_light");
    AiNodeSetStr(light, "name", "/obj/lit");

    logRenderError(AiRender(AI_RENDER_MODE_CAMERA));
    AiEnd();
    return 1;
}
