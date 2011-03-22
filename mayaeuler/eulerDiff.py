# import maya
# maya.cmds.loadPlugin("eulerDiff.py")
# maya.cmds.createNode("eulerDiff")

import math, sys

import maya.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMPx

kPluginNodeTypeName = "eulerDiff"

eulerNodeId = OpenMaya.MTypeId(0x2029000a)

# Node definition
class eulerNode(OpenMayaMPx.MPxNode):
	# class variables
	from_space = OpenMaya.MObject()
	to_space = OpenMaya.MObject()
	inTwist = OpenMaya.MObject()
	outputX = OpenMaya.MObject()
	outputY = OpenMaya.MObject()
	outputZ = OpenMaya.MObject()
	def __init__(self):
		OpenMayaMPx.MPxNode.__init__(self)
	def compute(self,plug,dataBlock):
		if ( plug == eulerNode.outputX or plug == eulerNode.outputY or plug == eulerNode.outputZ ):

			twist  = dataBlock.inputValue( eulerNode.inTwist ).asFloat()
			
			nrev = math.floor( twist/ 360.0 )
			frev = twist/ 360.0 - nrev
			
			if(frev > 0.5):
				nrev = nrev + 1
			
			#print "tw ", twist
			
			spaceA = dataBlock.inputValue( eulerNode.from_space ).asMatrix()
			spaceB = dataBlock.inputValue( eulerNode.to_space ).asMatrix()
			
			spaceA = spaceA * spaceB.inverse()
			
			tm = OpenMaya.MTransformationMatrix( spaceA )
			eul = tm.eulerRotation()
			
			#qua = eul.asQuaternion()
			
			#axis = OpenMaya.MVector()
			
			#scriptUtil = OpenMaya.MScriptUtil()
			
			#angptr = scriptUtil.asDoublePtr()
			
			#qua.getAxisAngle( axis, angptr)
			
			#ang = scriptUtil.getDouble(angptr)
			
			eul.reorder(OpenMaya.MEulerRotation.kZYX)
			
			#print "x ", eul.x, " y ", eul.y , " z ", eul.z
			
			resultX = -eul.x * 180.0 / 3.14159269 + nrev * 360.0
			outputHandle = dataBlock.outputValue( eulerNode.outputX )
			outputHandle.setFloat( resultX )
			
			resultY = eul.y * 180.0 / 3.14159269
			outputHandle = dataBlock.outputValue( eulerNode.outputY )
			outputHandle.setFloat( resultY )
			
			resultZ = eul.z * 180.0 / 3.14159269
			outputHandle = dataBlock.outputValue( eulerNode.outputZ )
			outputHandle.setFloat( resultZ )
			
			dataBlock.setClean( plug )

		return OpenMaya.kUnknownParameter

# creator
def nodeCreator():
	return OpenMayaMPx.asMPxPtr( eulerNode() )

# initializer
def nodeInitializer():
	# input
	nAttr = OpenMaya.MFnNumericAttribute()
	eulerNode.inTwist = nAttr.create( "inputTwist", "tiw", OpenMaya.MFnNumericData.kFloat, 0.0 )
	nAttr.setStorable(1)
	
	mAttr = OpenMaya.MFnMatrixAttribute()
	eulerNode.from_space = mAttr.create( "orginSpace", "orim" )
	eulerNode.to_space = mAttr.create( "destinationSpace", "destm" )
	
	# output
	nAttr = OpenMaya.MFnNumericAttribute()
	eulerNode.outputX = nAttr.create( "rotateX", "rx", OpenMaya.MFnNumericData.kFloat, 0.0 )
	nAttr.setStorable(1)
	nAttr.setWritable(1)
	
	eulerNode.outputY = nAttr.create( "rotateY", "ry", OpenMaya.MFnNumericData.kFloat, 0.0 )
	nAttr.setStorable(1)
	nAttr.setWritable(1)
	
	eulerNode.outputZ = nAttr.create( "rotateZ", "rz", OpenMaya.MFnNumericData.kFloat, 0.0 )
	nAttr.setStorable(1)
	nAttr.setWritable(1)
	
	# add attributes
	eulerNode.addAttribute( eulerNode.inTwist )
	eulerNode.addAttribute( eulerNode.from_space )
	eulerNode.addAttribute( eulerNode.to_space )
	eulerNode.addAttribute( eulerNode.outputX )
	eulerNode.addAttribute( eulerNode.outputY )
	eulerNode.addAttribute( eulerNode.outputZ )
	eulerNode.attributeAffects( eulerNode.inTwist, eulerNode.outputX )
	eulerNode.attributeAffects( eulerNode.from_space, eulerNode.outputX )
	eulerNode.attributeAffects( eulerNode.to_space, eulerNode.outputX )
	eulerNode.attributeAffects( eulerNode.from_space, eulerNode.outputY )
	eulerNode.attributeAffects( eulerNode.to_space, eulerNode.outputY )
	eulerNode.attributeAffects( eulerNode.from_space, eulerNode.outputZ )
	eulerNode.attributeAffects( eulerNode.to_space, eulerNode.outputZ )
	
# initialize the script plug-in
def initializePlugin(mobject):
	mplugin = OpenMayaMPx.MFnPlugin(mobject, "Zhang Jian", "0.1")
	try:
		mplugin.registerNode( kPluginNodeTypeName, eulerNodeId, nodeCreator, nodeInitializer )
	except:
		sys.stderr.write( "Failed to register node: %s" % kPluginNodeTypeName )
		raise

# uninitialize the script plug-in
def uninitializePlugin(mobject):
	mplugin = OpenMayaMPx.MFnPlugin(mobject, "Zhang Jian", "0.1")
	try:
		mplugin.deregisterNode( eulerNodeId )
	except:
		sys.stderr.write( "Failed to deregister node: %s" % kPluginNodeTypeName )
		raise
	
