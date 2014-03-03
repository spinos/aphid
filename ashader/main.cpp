#include <ai.h>

#include <cstdio>

extern AtNodeMethods* MayaMultiplyDivideMtd;
extern AtNodeMethods* MayaClampMtd;
extern AtNodeMethods* MayaGammaCorrectMtd;
extern AtNodeMethods* MayaConditionMtd;
extern AtNodeMethods* MayaReverseMtd;
extern AtNodeMethods* MayaBlendColorsMtd;
extern AtNodeMethods* MayaPlusMinusAverage1DMtd;
extern AtNodeMethods* MayaPlusMinusAverage2DMtd;
extern AtNodeMethods* MayaPlusMinusAverage3DMtd;
extern AtNodeMethods* MayaRemapValueToValueMtd;
extern AtNodeMethods* MayaRemapValueToColorMtd;
extern AtNodeMethods* MayaRemapColorMtd;
extern AtNodeMethods* MayaFileMtd;
extern AtNodeMethods* MayaPlace2DTextureMtd;
extern AtNodeMethods* MayaRampMtd;
extern AtNodeMethods* MayaProjectionMtd;
extern AtNodeMethods* MayaLuminanceMtd;
extern AtNodeMethods* MayaSetRangeMtd;
extern AtNodeMethods* MeshInfoMtd;
extern AtNodeMethods* MayaCheckerMtd;
extern AtNodeMethods* MayaBulgeMtd;
extern AtNodeMethods* MayaClothMtd;
extern AtNodeMethods* MayaGridMtd;
extern AtNodeMethods* MayaFractalMtd;
extern AtNodeMethods* MayaNoiseMtd;
extern AtNodeMethods* MayaPlace3DTextureMtd;
extern AtNodeMethods* MayaEnvSphereMtd;
extern AtNodeMethods* MayaRgbToHsvMtd;
extern AtNodeMethods* MayaHsvToRgbMtd;
extern AtNodeMethods* MayaLuminanceMtd;
extern AtNodeMethods* MayaCloudMtd;
extern AtNodeMethods* MayaCloudAlphaMtd;
extern AtNodeMethods* MayaSnowMtd;
extern AtNodeMethods* MayaContrastMtd;
extern AtNodeMethods* MayaLayeredTextureMtd;
extern AtNodeMethods* MayaLayeredShaderMtd;
extern AtNodeMethods* MayaSolidFractalMtd;
extern AtNodeMethods* MayaVolumeNoiseMtd;
extern AtNodeMethods* MayaBrownianMtd;
extern AtNodeMethods* MayaStuccoMtd;
extern AtNodeMethods* MayaRemapHsvMtd;
extern AtNodeMethods* MayaImagePlaneMtd;
extern AtNodeMethods* MayaSurfaceShaderMtd;
extern AtNodeMethods* ColorToFloatMtd;
extern AtNodeMethods* VectorToFloatMtd;
extern AtNodeMethods* PointToFloatMtd;
extern AtNodeMethods* Point2ToFloatMtd;
extern AtNodeMethods* WriteColorMtd;
extern AtNodeMethods* WriteFloatMtd;
extern AtNodeMethods* WriteColorInlineMtd;
extern AtNodeMethods* WriteFloatInlineMtd;
extern AtNodeMethods* WriteVectorInlineMtd;
extern AtNodeMethods* WritePointInlineMtd;
extern AtNodeMethods* WritePoint2InlineMtd;
extern AtNodeMethods* AnimMatrixMtd;
extern AtNodeMethods* AnimFloatMtd;
extern AtNodeMethods* AnimPointMtd;
extern AtNodeMethods* AnimVectorMtd;
extern AtNodeMethods* AnimColorMtd;
extern AtNodeMethods* UserDataFloatMtd;
extern AtNodeMethods* UserDataVectorMtd;
extern AtNodeMethods* UserDataColorMtd;
extern AtNodeMethods* UserDataStringMtd;
extern AtNodeMethods* UserDataBoolMtd;
extern AtNodeMethods* MayaShadingEngineMtd;
extern AtNodeMethods* SkinSssMethods;
extern AtNodeMethods* MayaSamplerInfo1DMtd;
extern AtNodeMethods* MayaSamplerInfo2DMtd;
extern AtNodeMethods* MayaSamplerInfo3DMtd;
extern AtNodeMethods* MayaVectorDisplacementMtd;
extern AtNodeMethods* MayaNormalDisplacementMtd;
extern AtNodeMethods* ShadowCatcherMtd;
extern AtNodeMethods* MayaHairMtd;
extern AtNodeMethods* MeshLightMaterialMtd;
extern AtNodeMethods* UserDataPnt2Mtd;
extern AtNodeMethods* UserDataIntMtd;
extern AtNodeMethods* MayaBump2DMtd;
extern AtNodeMethods* MayaFluidMtd;

enum{
   SHADER_MULTIPLYDIVIDE = 0,
   SHADER_CLAMP,
   SHADER_GAMMACORRECT,
   SHADER_CONDITION,
   SHADER_REVERSE,
   SHADER_BLENDCOLORS,
   SHADER_SAMPLERINFO1D,
   SHADER_SAMPLERINFO2D,
   SHADER_PLUSMINUSAVERAGE1D,
   SHADER_PLUSMINUSAVERAGE2D,
   SHADER_PLUSMINUSAVERAGE3D,
   SHADER_REMAPVALUETOVALUE,
   SHADER_REMAPVALUETOCOLOR,
   SHADER_REMAPCOLOR,
   SHADER_FILE,
   SHADER_PLACE2DTEXTURE,
   SHADER_RAMP,
   SHADER_PROJECTION,
   SHADER_MESHINFO,
   SHADER_CHECKER,
   SHADER_BULGE,
   SHADER_CLOTH,
   SHADER_GRID,
   SHADER_FRACTAL,
   SHADER_NOISE,
   SHADER_PLACE3DTEXTURE,
   SHADER_RGBTOHSV,
   SHADER_HSVTORGB,
   SHADER_LUMINANCE,
   SHADER_CLOUD,
   SHADER_CLOUDALPHA,
   SHADER_SNOW,
   SHADER_CONTRAST,
   SHADER_LAYEREDTEXTURE,
   SHADER_LAYEREDSHADER,
   SHADER_SOLIDFRACTAL,
   SHADER_VOLUMENOISE,
   SHADER_BROWNIAN,
   SHADER_STUCCO,
   SHADER_REMAPHSV,
   SHADER_SETRANGE,
   SHADER_IMAGEPLANE,
   SHADER_SURFACESHADER,
   SHADER_COLORTOFLOAT,
   SHADER_VECTORTOFLOAT,
   SHADER_POINTTOFLOAT,
   SHADER_POINT2TOFLOAT,
   SHADER_WRITECOLOR,
   SHADER_WRITECOLORINLINE,
   SHADER_WRITEFLOAT,
   SHADER_WRITEFLOATINLINE,
   SHADER_ENVSPHERE,
   SHADER_ANIMMATRIX,
   SHADER_ANIMFLOAT,
   SHADER_ANIMPOINT,
   SHADER_ANIMVECTOR,
   SHADER_ANIMCOLOR,
   SHADER_USERDATAFLOAT,
   SHADER_USERDATAVECTOR,
   SHADER_USERDATACOLOR,
   SHADER_USERDATASTRING,
   SHADER_SHADINGENGINE,
   SHADER_SKINSSS,
   SHADER_SAMPLERINFO3D,
   SHADER_WRITEVECTORINLINE,
   SHADER_WRITEPOINTINLINE,
   SHADER_WRITEPOINT2INLINE,
   SHADER_NORMALDISPLACEMENT,
   SHADER_VECTORDISPLACEMENT,
   SHADER_USERDATABOOL,
   SHADER_SHADOWCATCHER,
   SHADER_HAIR,
   SHADER_LIGHTMATERIAL,
   SHADER_USERDATAPNT2,
   SHADER_USERDATAINT,
   SHADER_BUMP2D,
   SHADER_MAYAFLUID
};

node_loader
{
   switch (i)
   {
   case SHADER_MULTIPLYDIVIDE:
      node->methods     = MayaMultiplyDivideMtd;
      node->output_type = AI_TYPE_RGB;
      node->name        = "MayaMultiplyDivide";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_CLAMP:
      node->methods     = MayaClampMtd;
      node->output_type = AI_TYPE_RGB;
      node->name        = "MayaClamp";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_GAMMACORRECT:
      node->methods     = MayaGammaCorrectMtd;
      node->output_type = AI_TYPE_RGB;
      node->name        = "MayaGammaCorrect";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_CONDITION:
      node->methods     = MayaConditionMtd;
      node->output_type = AI_TYPE_RGB;
      node->name        = "MayaCondition";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_REVERSE:
      node->methods     = MayaReverseMtd;
      node->output_type = AI_TYPE_RGB;
      node->name        = "MayaReverse";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_BLENDCOLORS:
      node->methods     = MayaBlendColorsMtd;
      node->output_type = AI_TYPE_RGB;
      node->name        = "MayaBlendColors";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_SAMPLERINFO1D:
      node->methods     = MayaSamplerInfo1DMtd;
      node->output_type = AI_TYPE_FLOAT;
      node->name        = "MayaSamplerInfo1D";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_SAMPLERINFO2D:
      node->methods     = MayaSamplerInfo2DMtd;
      node->output_type = AI_TYPE_POINT2;
      node->name        = "MayaSamplerInfo2D";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_PLUSMINUSAVERAGE1D:
      node->methods     = MayaPlusMinusAverage1DMtd;
      node->output_type = AI_TYPE_FLOAT;
      node->name        = "MayaPlusMinusAverage1D";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_PLUSMINUSAVERAGE2D:
      node->methods     = MayaPlusMinusAverage2DMtd;
      node->output_type = AI_TYPE_POINT2;
      node->name        = "MayaPlusMinusAverage2D";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_PLUSMINUSAVERAGE3D:
      node->methods     = MayaPlusMinusAverage3DMtd;
      node->output_type = AI_TYPE_RGB;
      node->name        = "MayaPlusMinusAverage3D";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_REMAPVALUETOVALUE:
      node->methods     = MayaRemapValueToValueMtd;
      node->output_type = AI_TYPE_FLOAT;
      node->name        = "MayaRemapValueToValue";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_REMAPVALUETOCOLOR:
      node->methods     = MayaRemapValueToColorMtd;
      node->output_type = AI_TYPE_RGB;
      node->name        = "MayaRemapValueToColor";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_REMAPCOLOR:
      node->methods     = MayaRemapColorMtd;
      node->output_type = AI_TYPE_RGB;
      node->name        = "MayaRemapColor";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_FILE:
      node->methods     = MayaFileMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaFile";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_PLACE2DTEXTURE:
      node->methods     = MayaPlace2DTextureMtd;
      node->output_type = AI_TYPE_POINT2;
      node->name        = "MayaPlace2DTexture";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_RAMP:
      node->methods     = MayaRampMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaRamp";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_PROJECTION:
      node->methods     = MayaProjectionMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaProjection";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_MESHINFO:
      node->methods     = MeshInfoMtd;
      node->output_type = AI_TYPE_RGB;
      node->name        = "MeshInfo";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_CHECKER:
      node->methods     = MayaCheckerMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaChecker";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_BULGE:
      node->methods     = MayaBulgeMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaBulge";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_CLOTH:
      node->methods     = MayaClothMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaCloth";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_GRID:
      node->methods     = MayaGridMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaGrid";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_FRACTAL:
      node->methods     = MayaFractalMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaFractal";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_NOISE:
      node->methods     = MayaNoiseMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaNoise";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_PLACE3DTEXTURE:
      node->methods     = MayaPlace3DTextureMtd;
      node->output_type = AI_TYPE_MATRIX;
      node->name        = "MayaPlace3DTexture";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_RGBTOHSV:
      node->methods     = MayaRgbToHsvMtd;
      node->output_type = AI_TYPE_VECTOR;
      node->name        = "MayaRgbToHsv";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_HSVTORGB:
      node->methods     = MayaHsvToRgbMtd;
      node->output_type = AI_TYPE_RGB;
      node->name        = "MayaHsvToRgb";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_LUMINANCE:
      node->methods     = MayaLuminanceMtd;
      node->output_type = AI_TYPE_FLOAT;
      node->name        = "MayaLuminance";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_CLOUD:
      node->methods     = MayaCloudMtd;
      node->output_type = AI_TYPE_RGB;
      node->name        = "MayaCloud";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_CLOUDALPHA:
      node->methods     = MayaCloudAlphaMtd;
      node->output_type = AI_TYPE_FLOAT;
      node->name        = "MayaCloudAlpha";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_SNOW:
      node->methods     = MayaSnowMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaSnow";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_CONTRAST:
      node->methods     = MayaContrastMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaContrast";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_LAYEREDTEXTURE:
      node->methods     = MayaLayeredTextureMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaLayeredTexture";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_LAYEREDSHADER:
      node->methods     = MayaLayeredShaderMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaLayeredShader";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_SOLIDFRACTAL:
      node->methods     = MayaSolidFractalMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaSolidFractal";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_VOLUMENOISE:
      node->methods     = MayaVolumeNoiseMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaVolumeNoise";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_BROWNIAN:
      node->methods     = MayaBrownianMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaBrownian";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_STUCCO:
      node->methods     = MayaStuccoMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaStucco";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_REMAPHSV:
      node->methods     = MayaRemapHsvMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaRemapHsv";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_SETRANGE:
      node->methods     = MayaSetRangeMtd;
      node->output_type = AI_TYPE_VECTOR;
      node->name        = "MayaSetRange";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_IMAGEPLANE:
      node->methods     = MayaImagePlaneMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaImagePlane";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_SURFACESHADER:
      node->methods     = MayaSurfaceShaderMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaSurfaceShader";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_COLORTOFLOAT:
      node->methods     = ColorToFloatMtd;
      node->output_type = AI_TYPE_FLOAT;
      node->name        = "colorToFloat";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_VECTORTOFLOAT:
      node->methods     = VectorToFloatMtd;
      node->output_type = AI_TYPE_FLOAT;
      node->name        = "vectorToFloat";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_POINTTOFLOAT:
      node->methods     = PointToFloatMtd;
      node->output_type = AI_TYPE_FLOAT;
      node->name        = "pointToFloat";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_POINT2TOFLOAT:
      node->methods     = Point2ToFloatMtd;
      node->output_type = AI_TYPE_FLOAT;
      node->name        = "point2ToFloat";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_WRITECOLOR:
      node->methods     = WriteColorMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "writeColor";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_WRITECOLORINLINE:
      node->methods     = WriteColorInlineMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "aovWriteColor";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_WRITEFLOAT:
      node->methods     = WriteFloatMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "writeFloat";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_WRITEFLOATINLINE:
      node->methods     = WriteFloatInlineMtd;
      node->output_type = AI_TYPE_FLOAT;
      node->name        = "aovWriteFloat";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_ENVSPHERE:
      node->methods     = MayaEnvSphereMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaEnvSphere";
      node->node_type   = AI_NODE_SHADER;
      break;      

   case SHADER_ANIMMATRIX:
      node->methods     = AnimMatrixMtd;
      node->output_type = AI_TYPE_MATRIX;
      node->name        = "anim_matrix";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_ANIMFLOAT:
      node->methods     = AnimFloatMtd;
      node->output_type = AI_TYPE_FLOAT;
      node->name        = "anim_float";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_ANIMPOINT:
      node->methods     = AnimPointMtd;
      node->output_type = AI_TYPE_POINT;
      node->name        = "anim_point";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_ANIMVECTOR:
      node->methods     = AnimVectorMtd;
      node->output_type = AI_TYPE_VECTOR;
      node->name        = "anim_vector";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_ANIMCOLOR:
      node->methods     = AnimColorMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "anim_color";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_USERDATAFLOAT:
      node->methods     = UserDataFloatMtd;
      node->output_type = AI_TYPE_FLOAT;
      node->name        = "userDataFloat";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_USERDATAVECTOR:
      node->methods     = UserDataVectorMtd;
      node->output_type = AI_TYPE_VECTOR;
      node->name        = "userDataVector";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_USERDATACOLOR:
      node->methods     = UserDataColorMtd;
      node->output_type = AI_TYPE_RGB;
      node->name        = "userDataColor";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_USERDATASTRING:
      node->methods     = UserDataStringMtd;
      node->output_type = AI_TYPE_STRING;
      node->name        = "userDataString";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_SHADINGENGINE:
      node->methods     = MayaShadingEngineMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaShadingEngine";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_SKINSSS:
      node->methods     = SkinSssMethods;
      node->output_type = AI_TYPE_RGB;
      node->name        = "skin_sss";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_SAMPLERINFO3D:
      node->methods     = MayaSamplerInfo3DMtd;
      node->output_type = AI_TYPE_RGB;
      node->name        = "MayaSamplerInfo3D";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_WRITEVECTORINLINE:
      node->methods     = WriteVectorInlineMtd;
      node->output_type = AI_TYPE_VECTOR;
      node->name        = "aovWriteVector";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_WRITEPOINTINLINE:
      node->methods     = WritePointInlineMtd;
      node->output_type = AI_TYPE_POINT;
      node->name        = "aovWritePoint";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_WRITEPOINT2INLINE:
      node->methods     = WritePoint2InlineMtd;
      node->output_type = AI_TYPE_POINT2;
      node->name        = "aovWritePoint2";
      node->node_type   = AI_NODE_SHADER;
      break;
      
   case SHADER_NORMALDISPLACEMENT:
      node->methods     = MayaNormalDisplacementMtd;
      node->output_type = AI_TYPE_FLOAT;
      node->name        = "MayaNormalDisplacement";
      node->node_type   = AI_NODE_SHADER;
      break;
      
   case SHADER_VECTORDISPLACEMENT:
      node->methods     = MayaVectorDisplacementMtd;
      node->output_type = AI_TYPE_VECTOR;
      node->name        = "MayaVectorDisplacement";
      node->node_type   = AI_NODE_SHADER;
      break;
      
   case SHADER_USERDATABOOL:
      node->methods     = UserDataBoolMtd;
      node->output_type = AI_TYPE_BOOLEAN;
      node->name        = "userDataBool";
      node->node_type   = AI_NODE_SHADER;
      break;
      
   case SHADER_SHADOWCATCHER:
      node->methods     = ShadowCatcherMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "shadowCatcher";
      node->node_type   = AI_NODE_SHADER;
      break;
      
   case SHADER_HAIR:
      node->methods     = MayaHairMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "MayaHair";
      node->node_type   = AI_NODE_SHADER;
      break;
      
   case SHADER_LIGHTMATERIAL:
      node->methods     = MeshLightMaterialMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "meshLightMaterial";
      node->node_type   = AI_NODE_SHADER;
      break;

   case SHADER_USERDATAPNT2:
      node->methods     = UserDataPnt2Mtd;
      node->output_type = AI_TYPE_POINT2;
      node->name        = "userDataPnt2";
      node->node_type   = AI_NODE_SHADER;
      break;
      
   case SHADER_USERDATAINT:
      node->methods     = UserDataIntMtd;
      node->output_type = AI_TYPE_INT;
      node->name        = "userDataInt";
      node->node_type   = AI_NODE_SHADER;
      break;
      
   case SHADER_BUMP2D:
      node->methods     = MayaBump2DMtd;
      node->output_type = AI_TYPE_RGBA;
      node->name        = "mayaBump2D";
      node->node_type   = AI_NODE_SHADER;
      break;
      
   case SHADER_MAYAFLUID:
      node->methods     = MayaFluidMtd;
      node->output_type = AI_TYPE_RGB;
      node->name        = "mayaFluid";
      node->node_type   = AI_NODE_SHADER;
      break;

   default:
      return false;
   }

   sprintf(node->version, AI_VERSION);

   return true;
}
