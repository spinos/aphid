#include "ai.h"
#include <sstream>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <vector>
using namespace std;

// UI parameters get stored into global vars
float UI_rootWidth;
float UI_tipWidth;
float UI_fuzz;
float UI_kink;
float UI_clumping;
int UI_hairCount;
char *UI_cachename;
float g_epsilon;

//utility global vars

int g_numBlocks = 20;
int g_blockSize;
int g_totalCurveCount = 0;
float g_averageArea;

// we read the UI parameters into their global vars
static int MyInit(AtNode *mynode, void **user_ptr)
{
   *user_ptr = mynode; // make a copy of the parent procedural   
   /*UI_rootWidth = AiNodeGetFlt(mynode, "rootWidth");
   UI_tipWidth = AiNodeGetFlt(mynode, "tipWidth");
   UI_fuzz = AiNodeGetFlt(mynode, "fuzz");
   UI_kink = AiNodeGetFlt(mynode, "kink");
   UI_clumping = AiNodeGetFlt(mynode, "clumping");
   UI_hairCount = AiNodeGetInt(mynode, "hairCount");
   UI_cachename = (char *)malloc(1024);
   UI_cachename = (char *)AiNodeGetStr(mynode, "cacheName");
   AtRGB rootColor = AiNodeGetRGB(mynode, "rootColor");
   AtRGB tipColor = AiNodeGetRGB(mynode, "tipColor");
   AtRGB mutantColor = AiNodeGetRGB(mynode, "mutantColor");
   float mutantScale = AiNodeGetFlt(mynode, "mutantScale");
   
   
   
   attrib = new HairAttribute;
   attrib->fuzz = UI_fuzz;
   attrib->kink = UI_kink;
   attrib->clumping = UI_clumping;
   attrib->rootMap = parseMap(mynode, "rootName");
   attrib->tipMap = parseMap(mynode, "tipName");
   attrib->fuzzMap = parseMap(mynode, "fuzzName");
   attrib->kinkMap = parseMap(mynode, "kinkName");
   attrib->clumpingMap = parseMap(mynode, "clumpingName");
   attrib->root_color = Vector3F(rootColor.r, rootColor.g, rootColor.b);
   attrib->tip_color = Vector3F(tipColor.r, tipColor.g, tipColor.b);
   attrib->mutant_color = Vector3F(mutantColor.r, mutantColor.g, mutantColor.b);
   attrib->mutant_scale = mutantScale;
   */
   return TRUE;
}

static int MyCleanup(void *user_ptr)
{
   return TRUE;
}

// we will create one node per chunk as set in the UI
static int MyNumNodes(void *user_ptr)
{
    return 1;
}

static AtNode *testSample(int blockIdx)
{
    AtNode *sphNode = AiNode("sphere");
    //AiNodeSetStr(curveNode, "basis", "catmull-rom");
   // AiNodeSetInt(curveNode, "visibility", 65523);
    //AiNodeSetInt(curveNode, "sidedness", 2);
    //AiNodeSetBool(curveNode, "self_shadows", 1);
    AiNodeSetFlt(sphNode, "radius",  1.13f);
    //AiNodeDeclare(curveNode, "uparamcoord", "uniform FLOAT");
    //AiNodeDeclare(curveNode, "vparamcoord", "uniform FLOAT");
    //AiNodeDeclare(curveNode, "hairrootcolor", "uniform RGB");
    //AiNodeDeclare(curveNode, "hairtipcolor", "uniform RGB");
	
	//AiNodeSetArray(curveNode, "num_points", counts);
	//AiNodeSetArray(curveNode, "points", points);
	//AiNodeSetArray(curveNode, "radius", radius);
	//AiNodeSetArray(curveNode, "hairrootcolor", rcolors);
	//AiNodeSetArray(curveNode, "hairtipcolor", tcolors);
	//AiNodeSetArray(curveNode, "uparamcoord", uparamcoord);
	//AiNodeSetArray(curveNode, "vparamcoord", vparamcoord);
	
	return sphNode;
}

// this builds the "points" node in Arnold and sets 
// the point locations, radius, and sets it to sphere mode
static AtNode *build_node(int i)
{
    return testSample(i);
}

// this is the function that Arnold calls to request the nodes
// that this procedural creates.
static AtNode *MyGetNode(void *user_ptr, int i)
{
   return build_node(i);
}

// vtable passed in by proc_loader macro define
proc_loader
{
   vtable->Init     = MyInit;
   vtable->Cleanup  = MyCleanup;
   vtable->NumNodes = MyNumNodes;
   vtable->GetNode  = MyGetNode;
   strcpy(vtable->version, AI_VERSION);
   return TRUE;
}