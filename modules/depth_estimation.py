import torch
from ai_models.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2

class DepthEstimation:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_model()
        
    def _init_model(self):
        """
        Initialize the DepthAnythingV2 model
        """
        print(f"Initializing Depth Estimation Model")
        self.model = DepthAnythingV2(**self.config.DEPTH_ANYTHING_MODEL_CONFIGS[self.config.DEPTH_ANYTHING_ENCODER])
        self.model.load_state_dict(torch.load(f'{self.config.DEPTH_ANYTHING_CHECKPOINT_DIR}/depth_anything_v2_{self.config.DEPTH_ANYTHING_ENCODER}.pth', map_location='cpu')) # TODO: NATHAN maybe change this to gpu?
        self.model = self.model.to(self.device).eval()

        
    def detect(self, image):
        """
        Return DepthAnythingV2 depth map for the given image
        """       

        #raw_img = cv2.imread('')
        depth = self.model.infer_image(image) # HxW raw depth map in numpy
        return depth