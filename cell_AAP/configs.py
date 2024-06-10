from cell_AAP import defaults
import skimage

class Cfg:
   
   "Cfg class to be passed to cell_AAP.annotation.Annotator"

   def __init__(
           self,
           version:float,
           threshold:skimage.filters,
           threshold_division: float,
           tophatstruct:skimage.morphology,
           erosionstruct:skimage.morphology,
           gaussian_sigma:float,
           pointprompts:bool,
           box_prompts:bool,
           propslist:list[str],
           frame_step:int,
           box_size:tuple
   ):
       self.VERSION = version
       self.threshold = threshold
       self.threshold_division = threshold_division
       self.tophatstruct = tophatstruct
       self.erosionstruct = erosionstruct
       self.gaussian_sigma = gaussian_sigma
       self.pointprompts = pointprompts
       self.box_prompts = box_prompts
       self.propslist = propslist
       self.frame_step = frame_step
       self.box_size = box_size


   @classmethod
   def get_config(cls, default:dict = defaults._DEFAULT):
       
       "Grabs a set of default configs from cell_AAP.defaults"

       try:
            return cls(
                version = default.VERSION,
                threshold = default.THRESHOLD,
                threshold_division = default.THRESHOLD_DIVISION,
                tophatstruct = default.TOPHATSTRUCT,
                erosionstruct = default.EROSIONSTRUCT,
                pointprompts = default.POINTPROMPTS,
                box_prompts = default.BOXPROMPTS,
                propslist = default.PROPSLIST,
                frame_step = default.FRAMESTEP,
                box_size = default.BOX_SIZE
            )
            
       except Exception as error:
           raise AttributeError(
               'argument passed to default must come from cell_AAP.defaults'
           )
       

            
       


       
       
   


       



                         



