#include <ai.h>
#include <cstdio>
#include <iostream>

extern AtNodeMethods* MallardFeatherUVMethod;

enum{
   SHADER_MALLARDFEATHERUV = 0
};

node_loader
{
   switch (i)
   {
   case SHADER_MALLARDFEATHERUV:
      node->methods     = MallardFeatherUVMethod;
      node->output_type = AI_TYPE_POINT2;
      node->name        = "featherUVCoord";
      node->node_type   = AI_NODE_SHADER;
      break;

   default:
      return false;
   }
   
   strcpy(node->version, AI_VERSION);
   return true;
}
