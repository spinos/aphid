#include <ai.h>

#define ARNOLD_NODEID_USER_DATA_PNT2            0x00115D1C

AI_SHADER_NODE_EXPORT_METHODS(UserDataPnt2Mtd);

namespace
{

enum UserDataPnt2Params
{
   p_pnt2AttrName,
   p_defaultValue
};
}

node_parameters
{
   AiMetaDataSetStr(mds, NULL, "maya.name", "aiUserDataPnt2");
   AiMetaDataSetInt(mds, NULL, "maya.id", ARNOLD_NODEID_USER_DATA_PNT2);
   AiMetaDataSetStr(mds, NULL, "maya.classification", "shader/utility");
   AiMetaDataSetBool(mds, NULL, "maya.swatch", false);

   AiParameterSTR("pnt2AttrName", "");
   AiParameterPNT2("defaultValue", 0.f, 0.f);
}

node_initialize
{
}

node_update
{
}

node_finish
{
}

shader_evaluate
{
   AtPoint2 p2;
   if (AiUDataGetPnt2(AiShaderEvalParamStr(p_pnt2AttrName), &p2))
   {
      sg->out.PNT2 = p2;
   }
   else
   {
      sg->out.PNT2 = AiShaderEvalParamPnt2(p_defaultValue);
   }
}

