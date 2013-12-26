#include <ai.h>

#include <iostream>
#include <vector>

int main(int argc, char *argv[])
{
    std::cout<<"hello\n";
    AiBegin();
    
    AtNode* options = AiNode("options");
    AtArray* outputs  = AiArrayAllocate(1, 1, AI_TYPE_STRING);
    AiArraySetStr(outputs, 0, "RGBA");
    //AiArraySetStr(outputs, 1, "RGBA");
    //AiArraySetStr(outputs, 2, "output:gaussian_filter");
    //AiArraySetStr(outputs, 3, "output:exr");
    AiNodeSetArray(options, "outputs", outputs);
    
    AtArray *arr = AiNodeGetArray(options, "outputs");
    std::cout<<"n elm "<<arr->nelements<<"\n";
    for(unsigned i = 0; i < arr->nelements; i++)
        std::cout<<" "<<AiArrayGetStr(arr, i);
    
    AiNodeSetInt(options, "xres", 320);
    AiNodeSetInt(options, "yres", 240);
    AiNodeSetPtr(options, "camera", "/obj/cam");
    AiNodeSetInt(options, "AA_samples", 3);
    
    AtNode* driver = AiNode("driver_exr");
    AiNodeSetStr(driver, "name", "output:exr");
    AiNodeSetStr(driver, "filename", "output.exr");
    AiNodeSetFlt(driver, "gamma", 2.2f);
    
    AtNode * camera = AiNode("persp_camera");
    AiNodeSetStr(camera, "name", "/obj/cam");
    //AiNodeSetFlt(camera, "fov", 54.f);
    //AiNodeSetFlt(camera, "near_clip", 0.100000001);
    //AiNodeSetFlt(camera, "far_clip", 10000);
    AtMatrix matrix;
    //AiM4Identity(matrix);
    
    AiNodeGetMatrix(camera, "matrix", matrix);
    
    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++)
        std::cout<<"m["<<i<<"]["<<j<<"] = "<<matrix[i][j]<<"\n";
    
    
    //
    //
    //AiNodeSetMatrix(camera, "matrix", matrix);
    /*
    //AtNode * filter = AiNode("gaussian_filter");
    //AiNodeSetStr(filter, "name", "output:gaussian_filter");
    
    

    AtNode * standard = AiNode("standard");
    AiNodeSetStr(standard, "name", "/shop/standard1");

    AtNode * sphere = AiNode("sphere");
    AiNodeSetStr(sphere, "shader", "/shop/standard1");
    
    AtNode * light = AiNode("point_light");
    AiNodeSetStr(light, "name", "/obj/lit");
    
    AtNode * se = AiNode("MayaShadingEngine");
    AiNodeSetStr(se, "name", "initialShadingGroup");
    AiNodeSetStr(se, "beauty", "lambert1");

    AtNode * lambert = AiNode("lambert");
    AiNodeSetStr(lambert, "name", "lambert1");
*/

    int ai_status(AI_SUCCESS);
    ai_status = AiRender(AI_RENDER_MODE_CAMERA);
    if (ai_status != AI_SUCCESS) {
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
    AiEnd();
    return 1;
}
