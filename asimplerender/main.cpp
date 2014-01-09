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

void logRenderError(int status)
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

int main(int argc, char *argv[])
{
    AiBegin();
    
    AiLoadPlugins("./driver_foo.dll");
    
    AtNode* options = AiNode("options");
    AtArray* outputs  = AiArrayAllocate(1, 1, AI_TYPE_STRING);
    AiArraySetStr(outputs, 0, "RGBA RGBA output:gaussian_filter output/foo");
    AiNodeSetArray(options, "outputs", outputs);
    
    AiNodeSetInt(options, "xres", 400);
    AiNodeSetInt(options, "yres", 300);
    AiNodeSetInt(options, "AA_samples", 3);
	
    AtNode* driver = AiNode("driver_jpeg");
    AiNodeSetStr(driver, "name", "output/foo");
    AiNodeSetStr(driver, "filename", "output.jpg");
    //AiNodeSetFlt(driver, "gamma", 2.2f);
    
    AtNode * camera = AiNode("persp_camera");
    AiNodeSetStr(camera, "name", "/obj/cam");
    AiNodeSetFlt(camera, "fov", 35.f);
    AiNodeSetFlt(camera, "near_clip", 0.100000001);
    AiNodeSetFlt(camera, "far_clip", 1000000);
    AtMatrix matrix;
    AiM4Identity(matrix);
    //matrix[0][0] = 0.f;
    //matrix[0][1] = 0.f;
    //matrix[0][2] = 1.f;
    //matrix[2][0] = -1.f;
    //matrix[2][0] = 0.f;
    //matrix[2][1] = 0.f;
    matrix[3][2] = 15.f;
	AiNodeSetMatrix(camera, "matrix", matrix);
	
    AiNodeGetMatrix(camera, "matrix", matrix);
    
	AiNodeSetPtr(options, "camera", camera);
    
	AtNode * filter = AiNode("gaussian_filter");
    AiNodeSetStr(filter, "name", "output:gaussian_filter");

    AtNode * standard = AiNode("standard");
    AiNodeSetStr(standard, "name", "/shop/standard1");
    AiNodeSetRGB(standard, "Kd_color", 1, 0, 0);

    //AtNode * sphere = AiNode("sphere");
    //AiNodeSetPtr(sphere, "shader", standard);
    
    AtNode * light = AiNode("point_light");
    AiNodeSetStr(light, "name", "/obj/lit");
    AiNodeSetFlt(light, "intensity", 1024);
    matrix[3][0] = -10.f;
    AiNodeSetMatrix(light, "matrix", matrix);
    
    AtNode *curveNode = AiNode("curves");
    AiNodeSetStr(curveNode, "basis", "catmull-rom");
    AtArray* counts = AiArrayAllocate(1, 1, AI_TYPE_UINT);
    AiArraySetUInt(counts, 0, 4 + 2);
    
    AtArray* points = AiArrayAllocate(4 + 2, 1, AI_TYPE_POINT);
    
    AtPoint pt;
    pt.x = 0.f;
    pt.y = 0.f;
    pt.z = 0.f;
    AiArraySetPnt(points, 0, pt);
    AiArraySetPnt(points, 1, pt);
    pt.x = 0.f;
    pt.y = 1.f;
    AiArraySetPnt(points, 2, pt);
    pt.x = 1.f;
    pt.y = 1.f;
    AiArraySetPnt(points, 3, pt);
    pt.x = 1.f;
    pt.y = 2.f;
    AiArraySetPnt(points, 4, pt);
    AiArraySetPnt(points, 5, pt);
    
    AtArray* radius = AiArrayAllocate(4, 1, AI_TYPE_FLOAT);
    AiArraySetFlt(radius, 0, 0.04);
    AiArraySetFlt(radius, 1, 0.02);
    AiArraySetFlt(radius, 2, 0.02);
    AiArraySetFlt(radius, 3, 0.01);
    
    AiNodeSetArray(curveNode, "num_points", counts);
	AiNodeSetArray(curveNode, "points", points);
	AiNodeSetArray(curveNode, "radius", radius);
	
	AiNodeDeclare(curveNode, "color", "varying RGB");
	AtArray* colors = AiArrayAllocate(4, 1, AI_TYPE_RGB);
	
	AtRGB acol;
	acol.r = 1.f;
	acol.g = 1.f;
	acol.b = 1.f;
	
	AiArraySetRGB(colors, 0, acol);
	acol.r = .3f;
	AiArraySetRGB(colors, 1, acol);
	acol.r = .6f;
	acol.g = .7f;
	AiArraySetRGB(colors, 2, acol);
	acol.g = .9f;
	acol.b = .1f;
	AiArraySetRGB(colors, 3, acol);
	
	AiNodeSetArray(curveNode, "color", colors);
	
	AtNode *hair = AiNode("MayaHair");
	
	AiNodeSetRGB(hair, "hairColor", 1, 1, 1);
	
	AtArray* cs = AiArrayAllocate(256, 1, AI_TYPE_RGB);
	for(int i = 0; i < 256; i++) {
	    acol.r = .5f + (float)i/256 * .5f;
	    acol.g = acol.r;
	    acol.b = acol.r;
	    AiArraySetRGB(cs, i, acol);
	}
	
	AiNodeSetArray(hair, "hairColorScale", cs);
	
	AiNodeSetPtr(curveNode, "shader", hair);
    
    logRenderError(AiRender(AI_RENDER_MODE_CAMERA));
    AiEnd();
    return 1;
}
