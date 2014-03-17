import maya.cmds as cmds
import maya.mel as mel

class BasePluginOps(object):
    def __init__(self):
        '''base plugin operations'''
        pass
    
    def is_plugin_loaded(self, pluginName):
        return cmds.pluginInfo( pluginName, query=True, loaded=True )
        
    def load_plugin(self, pluginName):
        if self.is_plugin_loaded(pluginName):
            return True
        try:
            cmds.loadPlugin(pluginName, quiet=True)
            cmds.pluginInfo(pluginName, edit=True, autoload=True)
            return True
        except:
            return False
            
class BaseRenderGlobalOps(BasePluginOps):
    FormatAsInt = {'tif':3, 'iff':7, 'jpg':8, 'tga':19, 'png':32, 'exr':51}
    FormatAsStr = {3:'tif', 7:'iff', 8:'jpg', 19:'tga', 32:'png', 51:'exr'}
    def __init__(self, pluginName='Mayatomr', rendererName='mentalRay'):
        '''setup plugin renderer'''
        if self.load_plugin(pluginName):
            cmds.setAttr('defaultRenderGlobals.ren', rendererName, type='string')
        else:
            print('ERROR: %s is not available' % rendererName)
        cmds.setAttr('defaultRenderGlobals.animation', 1)
        cmds.setAttr('defaultRenderGlobals.outFormatControl', 0)
        cmds.setAttr('defaultRenderGlobals.fieldExtControl', 2)
        cmds.setAttr('defaultRenderGlobals.putFrameBeforeExt', 1)
        cmds.setAttr('defaultRenderGlobals.extensionPadding', 4)
        
    def renderer(self):
        return cmds.getAttr('defaultRenderGlobals.ren')
            
    def set_image_prefix(self, imagePrefix):
        if imagePrefix == '':
            return
        cmds.setAttr('defaultRenderGlobals.imageFilePrefix', imagePrefix, type='string')
        
    def image_prefix(self):
        return cmds.setAttr('defaultRenderGlobals.imageFilePrefix', type='string') 
        
    def set_image_size(self, width, height):
        cmds.setAttr('defaultResolution.width', width)
        cmds.setAttr('defaultResolution.height', height)
        
    def image_size(self):
        return (cmds.getAttr('defaultResolution.width'), cmds.setAttr('defaultResolution.height'))
        
    def use_camera(self, cameraName='perspShape'):
        for cam in cmds.listCameras():
            camattr = cam + '.renderable'
            cmds.setAttr(camattr, 0)
            
        cmds.setAttr((cameraName + '.renderable'), 1)
        
    def set_image_format(self, imageFormat):
        cmds.setAttr('defaultRenderGlobals.imageFormat', self.FormatAsInt[imageFormat])
        cmds.setAttr('defaultRenderGlobals.outFormatExt', imageFormat, type='string')
        
    def image_format(self):
        fmt = cmds.getAttr('defaultRenderGlobals.imageFormat')
        return self.FormatAsStr[fmt]
        
    def set_frame_range(self, startFrame, endFrame):
        if endFrame <= startFrame:
            startFrame = cmds.playbackOptions(query=True, min=True)
            endFrame = cmds.playbackOptions(query=True, max=True)
            
        cmds.setAttr('defaultRenderGlobals.startFrame', startFrame)
        cmds.setAttr('defaultRenderGlobals.endFrame', endFrame)
        
    def frame_range(self):
        start = int(cmds.getAttr('defaultRenderGlobals.startFrame'))
        end = int(cmds.getAttr('defaultRenderGlobals.endFrame'))
        return (start, end)

class ArnoldRenderOps(BaseRenderGlobalOps):
    def __init__(self, pluginName='mtoa', 
                        rendererName='arnold', 
                        imagePrefix='', 
                        imageWidth=2048, 
                        imageHeight=848, 
                        cameraName='perspShape', 
                        imageFormat='png', 
                        startFrame=0, 
                        endFrame=0):
        '''default mtoa settings'''
        BaseRenderGlobalOps.__init__(self, pluginName, rendererName)
        self.set_image_prefix(imagePrefix)
        self.set_image_size(imageWidth, imageHeight)
        self.use_camera(cameraName)
        self.set_image_format(imageFormat)
        self.set_frame_range(startFrame, endFrame)
        if self.is_arnold_ready():
            self.create_ai_options()
            cmds.setAttr('defaultArnoldRenderOptions.motion_blur_enable', 1)
        
    def is_arnold_ready(self):
        return self.renderer() == 'arnold'
        
    def create_ai_options(self):
        if not cmds.objExists('defaultArnoldRenderOptions'):
            print('create default aiOptions')
            cmds.createNode('aiOptions', name='defaultArnoldRenderOptions', skipSelect=True)

# a = ArnoldRenderOps(pluginName='Mayatomr', rendererName='mentalRay', imagePrefix='test')
# a = ArnoldRenderOps(imagePrefix='foo')

