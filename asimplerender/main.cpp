#include <ai.h>
#include "SimpleRenderConfig.h"
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

void testTex()
{
    AtNode * mesh = AiNode("polymesh");
    AiNodeSetStr(mesh, "name", "/obj/plane");
    
    AtArray * fcounts = AiArrayAllocate(1, 1, AI_TYPE_UINT);
    AiArraySetUInt(fcounts, 0, 4);
    
    AtArray * points = AiArrayAllocate(4, 1, AI_TYPE_POINT);
    AtPoint pt;
    pt.x = -1.1f;
    pt.y = 0.f;
    pt.z = -.5f;
    AiArraySetPnt(points, 0, pt);
    pt.x = .9f;
    pt.y = 0.f;
    pt.z = -.5f;
    AiArraySetPnt(points, 1, pt);
    pt.x = .9f;
    pt.y = 1.9f;
    pt.z = -.5f;
    AiArraySetPnt(points, 2, pt);
    pt.x = -1.1f;
    pt.y = 1.9f;
    pt.z = -.5f;
    AiArraySetPnt(points, 3, pt);
    
    AtArray * normals = AiArrayAllocate(4, 1, AI_TYPE_VECTOR);
    AtVector nor; nor.x = 0.f; nor.y =0.f; nor.z = 1.f;
    
    AiArraySetVec(normals, 0, nor);
    AiArraySetVec(normals, 1, nor);
    AiArraySetVec(normals, 2, nor);
    AiArraySetVec(normals, 3, nor);
    
    AtArray * vertices = AiArrayAllocate(4, 1, AI_TYPE_UINT);
    AiArraySetUInt(vertices, 0, 0);
    AiArraySetUInt(vertices, 1, 1);
    AiArraySetUInt(vertices, 2, 2);
    AiArraySetUInt(vertices, 3, 3);
    
    AtArray * nvertices = AiArrayAllocate(4, 1, AI_TYPE_UINT);
    AiArraySetUInt(nvertices, 0, 0);
    AiArraySetUInt(nvertices, 1, 1);
    AiArraySetUInt(nvertices, 2, 2);
    AiArraySetUInt(nvertices, 3, 3);
    
    AtArray * uvvertices = AiArrayAllocate(4, 1, AI_TYPE_UINT);
    AiArraySetUInt(uvvertices, 0, 0);
    AiArraySetUInt(uvvertices, 1, 1);
    AiArraySetUInt(uvvertices, 2, 2);
    AiArraySetUInt(uvvertices, 3, 3);
    
    AtArray * uvs = AiArrayAllocate(4, 1, AI_TYPE_POINT2);
    AtPoint2 auv; 
    auv.x = 0.f; auv.y = 0.f;
    AiArraySetPnt2(uvs, 0, auv);
    auv.x = 1.f; auv.y = 0.f;
    AiArraySetPnt2(uvs, 1, auv);
    auv.x = 1.f; auv.y = 1.f;
    AiArraySetPnt2(uvs, 2, auv);
    auv.x = 0.f; auv.y = 1.f;
    AiArraySetPnt2(uvs, 3, auv);
    
    AiNodeSetArray(mesh, "nsides", fcounts);
    AiNodeSetArray(mesh, "vidxs", vertices);
    AiNodeSetArray(mesh, "nidxs", nvertices);
    AiNodeSetArray(mesh, "uvidxs", uvvertices);
    AiNodeSetArray(mesh, "vlist", points);
    AiNodeSetArray(mesh, "nlist", normals);
    AiNodeSetArray(mesh, "uvlist", uvs);
    
    AtNode * stnd = AiNode("standard");
	AiNodeSetFlt(stnd, "Kd", .9f);

	AtNode * img = AiNode("image");
	AiNodeSetStr(img, "filename", "C:/Users/zhangjian/Desktop/LordsBird03.jpg");
	if(AiNodeLink(img, "Kd_color", stnd)) std::clog<<"kd color linked";
	else std::clog<<"kd color not linked";
	AiNodeSetPtr(mesh, "shader", stnd);
}

int main(int argc, char *argv[])
{
    std::clog<<"CMake Tutorial Version "<<SimpleRender_VERSION_MAJOR<<"."<<SimpleRender_VERSION_MINOR;
    
    AiBegin();
    
    AiLoadPlugins("./driver_foo.dll");
    AiLoadPlugins("./ExtendShaders.dll");
    
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
    const int nlines = 99;
    AtArray* counts = AiArrayAllocate(nlines, 1, AI_TYPE_UINT);
    for(int i = 0 ; i < nlines; i++) AiArraySetUInt(counts, i, 4 + 2);
    
    AtArray* points = AiArrayAllocate((4 + 2)*nlines, 1, AI_TYPE_POINT);
    
    AtPoint pt;
    int offset = 0;
    for(int i = 0 ; i < nlines; i++) {
        offset = i * 6;
        pt.x = 0.02f * i;
        pt.y = 0.f;
        pt.z = 0.f;
        AiArraySetPnt(points, 0 + offset, pt);
        AiArraySetPnt(points, 1 + offset, pt);
        pt.x = 0.02f * i;
        pt.y = 1.f;
        pt.z = 1.f;
        AiArraySetPnt(points, 2 + offset, pt);
        pt.x = 1.f + 0.02f * i;
        pt.y = 2.f;
        pt.z = 1.f;
        AiArraySetPnt(points, 3 + offset, pt);
        pt.x = 1.f + 0.02f * i;
        pt.y = 3.f;
        pt.z = 0.f;
        AiArraySetPnt(points, 4 + offset, pt);
        AiArraySetPnt(points, 5 + offset, pt);
    }
    
    AtArray* radius = AiArrayAllocate(4 * nlines, 1, AI_TYPE_FLOAT);
    for(int i = 0 ; i < nlines; i++) {
        offset = i * 4;
        AiArraySetFlt(radius, 0+ offset, 0.016);
        AiArraySetFlt(radius, 1+ offset, 0.015);
        AiArraySetFlt(radius, 2+ offset, 0.015);
        AiArraySetFlt(radius, 3+ offset, 0.014);
    }
    
    AiNodeSetArray(curveNode, "num_points", counts);
	AiNodeSetArray(curveNode, "points", points);
	AiNodeSetArray(curveNode, "radius", radius);
	
	AiNodeDeclare(curveNode, "colors", "varying RGB");
	AtArray* colors = AiArrayAllocate(4 * nlines, 1, AI_TYPE_RGB);
	
	AtRGB acol;
	for(int i = 0 ; i < nlines; i++) {
        offset = i * 4;
        acol.r = 0.f;
        acol.g = 1.f;
        acol.b = 0.f;
        
        AiArraySetRGB(colors, 0 + offset, acol);
        acol.r = .1f;
        acol.g = .1f;
        acol.b = 1.f;
        AiArraySetRGB(colors, 1 + offset, acol);
        acol.r = .99f;
        acol.g = .1f;
        acol.b = .0f;
        AiArraySetRGB(colors, 2 + offset, acol);
        acol.r = .1f;
        acol.g = .99f;
        acol.b = .1f;
        AiArraySetRGB(colors, 3 + offset, acol);
    }
	
	AiNodeSetArray(curveNode, "colors", colors);
	
	AiNodeDeclare(curveNode, "uvparamcoord", "varying POINT2");
 
    AtArray* uvcoords = AiArrayAllocate(4 * nlines, 1, AI_TYPE_POINT2);
    AtPoint2 auv; 
    for(int i = 0 ; i < nlines; i++) {
	    offset = 4 * i;
	    auv.x = 0.01f * i;
	    auv.y = 0.01f;
	    AiArraySetPnt2(uvcoords, 0 + offset, auv);
	    auv.y = 0.33f;
	    AiArraySetPnt2(uvcoords, 1 + offset, auv);
	    auv.y = 0.66f;
	    AiArraySetPnt2(uvcoords, 2 + offset, auv);
	    auv.y = 0.99f;
	    AiArraySetPnt2(uvcoords, 3 + offset, auv);
	}
	
	AiNodeSetArray(curveNode, "uvparamcoord", uvcoords);

	/*AtNode *hair = AiNode("hair");
	AiNodeSetFlt(hair, "gloss", 20);
	AiNodeSetFlt(hair, "gloss2", 8);
	AiNodeSetFlt(hair, "spec", 0.f);
	AiNodeSetFlt(hair, "spec2", 0.f);*/
	
	AtNode *hair = AiNode("utility");
	
	const AtNodeEntry* nodeEntry = AiNodeEntryLookUp("userDataColor");
	if(nodeEntry != NULL) std::clog<<"userDataColor exists";
	
	//AtNode * usrCol = AiNode("userDataColor");
	//AiNodeSetStr(usrCol, "colorAttrName", "colors");
	
	/*if(AiNodeLink(usrCol, "rootcolor", hair)) std::clog<<"linked";
	if(AiNodeLink(usrCol, "tipcolor", hair)) std::clog<<"linked";
	if(AiNodeLink(usrCol, "spec_color", hair)) std::clog<<"linked";
	*/
	
	AtNode * img = AiNode("image");
	AiNodeSetStr(img, "filename", "C:/Users/zhangjian/Desktop/LordsBird03.jpg");
	
	//if(AiNodeLink(usrCol, "color", hair)) std::clog<<"\ncolor as color linked";
	if(AiNodeLink(img, "color", hair)) std::clog<<"\nimage as color linked";
	
	AtNode * usrUV = AiNode("featherUVCoord");
	AiNodeSetStr(usrUV, "attrName", "uvparamcoord");
	if(AiNodeLink(usrUV, "uvcoords", img)) std::clog<<"\nuv coords linked";

	AiNodeSetPtr(curveNode, "shader", hair);
	
	//testTex();
	for(int i=1; i< 4; i++) {
        AtNode * ball = AiNode("sphere");
        
        AtNode * inst = AiNode("ginstance");
        AiNodeSetPtr(inst, "node", ball);
        
        AtMatrix mm;
        AiM4Identity(mm);
        mm[3][1] = i * 1.f;
        
        AtArray* mms = AiArrayAllocate(1, 1, AI_TYPE_MATRIX);
        AiArraySetMtx(mms, 0, mm);
        AiNodeSetArray(inst, "matrix", mms);
    }
    
    AtNode * arch = AiNode("procedural");
    AiNodeSetStr(arch, "dso", "D:/ofl/proxyPaint/plugins/spin.1.ass");
    AiNodeSetPnt(arch, "min", -3.11693668, 0.36803484, -1.12230337);
    AiNodeSetPnt(arch, "max", 3.13341951, 6.34418392, -0.452006936);
        
    logRenderError(AiRender(AI_RENDER_MODE_CAMERA));
    AiEnd();
    return 1;
}
