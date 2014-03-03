#include <ai.h>

AI_SHADER_NODE_EXPORT_METHODS(MallardFeatherUVMethod);

namespace
{
enum UserDataPnt2Params
{
   p_attrName,
   p_defaultValue
};
}

node_parameters
{
   AiParameterSTR("attrName", "");
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
   if (AiUDataGetPnt2(AiShaderEvalParamStr(p_attrName), &p2))
   {
      sg->out.PNT2 = p2;
   }
   else
   {
      sg->out.PNT2 = AiShaderEvalParamPnt2(p_defaultValue);
   }
}

