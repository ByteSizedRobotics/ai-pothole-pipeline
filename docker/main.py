import torch
import cv2
import asyncio
import websockets
import json
import threading
import numpy as np
import time
import os
import torch.nn as nn
from aiortc import RTCPeerConnection, RTCSessionDescription
from aimodels.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2

from torchvision import transforms as T
import aimodels.DeepLabV3Plus.network as network
import aimodels.DeepLabV3Plus.utils as utils
from aimodels.DeepLabV3Plus.datasets import VOCSegmentation, Cityscapes

class PotholeDetectionService:
    def __init__(self, model_path='aimodels/pothole_model_2025_03_01', webrtc_uri="ws://100.85.202.20:8765"): # TODO: NATHAN change this to raspi IP address (pass it into the docker container upon creation)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # TODO: NATHAN make sure GPU works
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path).to(self.device)
        self.webrtc_uri = webrtc_uri
        
        # Global variables for video stream
        self.current_frame = None
        self.pc = None
        self.new_pothole_detection = False
        self.latest_detection_frame = None
        self.latest_detections = []  # Store latest detections
        self.detection_count = 0
        
        print(f"Pothole detection service initialized on device: {self.device}")

    async def connect_to_webrtc(self):
        """Connect to ROS2 WebRTC publisher and receive video stream"""
        try:
            # Connect to WebRTC signaling server
            async with websockets.connect(self.webrtc_uri) as websocket:
                print("Connected to WebRTC signaling server")
                
                # Create peer connection
                self.pc = RTCPeerConnection()
                
                @self.pc.on("track")
                def on_track(track):
                    print(f"Track received: {track.kind}")
                    if track.kind == "video":
                        asyncio.create_task(self.receive_video_frames(track))

                # Create offer
                self.pc.addTransceiver("video", direction="recvonly")
                offer = await self.pc.createOffer()
                await self.pc.setLocalDescription(offer)
                
                # Send offer to ROS2 node
                await websocket.send(json.dumps({
                    "type": "offer",
                    "sdp": self.pc.localDescription.sdp
                }))
                
                # Wait for answer
                async for message in websocket:
                    data = json.loads(message)
                    if data.get("type") == "answer":
                        answer = RTCSessionDescription(sdp=data["sdp"], type="answer")
                        await self.pc.setRemoteDescription(answer)
                        print("WebRTC connection established")
                        break
                    elif data.get("type") == "ice-candidate":
                        # Handle ICE candidates if needed
                        pass
                                    
                # Keep connection alive
                await asyncio.Future()  # Run forever
                
        except Exception as e:
            print(f"WebRTC connection error: {e}")

    async def receive_video_frames(self, track):
        """Process incoming video frames from WebRTC stream"""
        while True:
            try:
                frame = await track.recv()
                # Convert av.VideoFrame to numpy array
                img = frame.to_ndarray(format="bgr24")
                self.current_frame = img.copy()
                
                # Process frame for pothole detection
                self.process_frame_for_detection(self.current_frame)
                
            except Exception as e:
                print(f"Error receiving frame: {e}")
                break

    def process_frame_for_detection(self, frame):
        """Process a single frame for pothole detection"""
        if frame is not None:
            # Run detection
            results = self.model(frame)
            detections = []
            for det in results.xyxy[0].tolist():
                x1, y1, x2, y2, conf, cls = det
                if conf >= 0.5:
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf)
                    })

                    # TODO: NATHAN don't draw rectangles on image in prod, only for debugging
                    # Draw rectangle and label on image
                    cv2.putText(frame, f"Detection {len(detections)}: {conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            # If potholes detected, save the frame
            if detections:
                self.new_pothole_detection = True
                self.latest_detection_frame = frame.copy()
                self.latest_detections = detections  # Store detections for queue
                self.detection_count += 1
                # TODO: NATHAN MAKE API ENDPOINT CALL TO send detected pothole image (make sure to send timestamp)
                timestamp = int(time.time())

                # TODO: NATHAN DEBUG CODE
                filename = f"pothole_detection_{self.detection_count}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Pothole detected! Saved frame as {filename} (Total detections: {self.detection_count})")

    def start_webrtc_connection(self):
        """Start WebRTC connection in background thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.connect_to_webrtc())

    def run(self):
        """Start the pothole detection service"""
        print("Starting pothole detection service...")
        print("Connecting to WebRTC stream and processing frames...")
        
        # Start WebRTC connection
        self.start_webrtc_connection()

class SeverityCalculationService():
    def __init__(self, frame, detections, resolution=(3280, 2464)):
        self.detections = detections
        self.resolution = resolution
        self.image = frame
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #initialize depth estimation model
        self.depth_model = DepthAnythingV2({'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}) # large model
        self.depth_model.load_state_dict(torch.load('aimodels/DepthAnythingV2/checkpoints/depth_anything_v2_vitl.pth', map_location='cpu')) # TODO: NATHAN change this to gpu?
        self.depth_model = self.depth_model.to(self.device).eval()

    def estimate_area(self):
        pothole_areas = []

        for detection in self.detections:
            bbox = detection['bbox']  # Extract bbox from dictionary
            x1, y1, x2, y2 = bbox
            bounding_box_area = (x2-x1)*(y2-y1)
            y_distance_middle_pothole = (y1+y2)/2
            x_distance_middle_pothole = (x1+x2)/2

            # TODO: NATHAN update this to have the different supported resolutions we are planning to use
            # TODO: NATHAN need to calculate the other scaling factors for the other resolutions
            if self.resolution == (3280, 2464):
                a = -0.00022788150560381466
                b = -249.57544427170112
                c = 0.6036184827412467
                d = -0.00036558500089469364
                
                y_scaling_factor = a/(b + c*y_distance_middle_pothole + d*(y_distance_middle_pothole**2))

                if x_distance_middle_pothole <= 1640:
                    x_scaling_factor = 4*(10**(-9))*x_distance_middle_pothole+6*(10**(-6))
                elif x_distance_middle_pothole > 1640:
                    x_scaling_factor = -4*(10**(-9))*x_distance_middle_pothole+2*(10**(-5))

            
            area = x_scaling_factor * y_scaling_factor * bounding_box_area * 100000
            pothole_areas.append(area)

        return pothole_areas
    
    def estimate_depth(self, image, pothole_areas):
        pothole_depths = []
        cropped_potholes = []
        
        for i, detection in enumerate(self.detections):
            bbox = detection['bbox']  # Extract bbox from dictionary
            x1, y1, x2, y2 = bbox
            
            x1, y1 = max(0, int(x1)), max(0, int(y1)) # make sure x1 and y1 are not negative
            x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2)) # make sure x2 and y2 are within image bounds
            
            cropped_pothole = image[y1:y2, x1:x2]
            
            # Check if cropped pothole is valid
            if cropped_pothole.size == 0:
                print(f"Warning: Empty cropped pothole for bbox {bbox}")
                pothole_depths.append(0.0)
                continue

            # Resize the cropped pothole to the target size
            resized_pothole = cv2.resize(cropped_pothole, (512, 256), interpolation=cv2.INTER_AREA)
            cropped_potholes.append(resized_pothole)

            # Use the correct model reference and method
            depth_map = self.depth_model(resized_pothole)  # perform depth estimation with DepthAnythingV2 model

            min_depth = np.percentile(depth_map, 50)
            max_depth = np.percentile(depth_map, 95)

            relative_depth = max_depth - min_depth

            # Normalize the depth by dividing the relative depth by the square root of the area
            # Or else bigger potholes will have higher depth values than smaller potholes 
            # regardless of the actual depth of the pothole.
            normalized_depth = (relative_depth / np.sqrt(pothole_areas[i])) / 1000
            pothole_depths.append(normalized_depth)
            
        return pothole_depths
    
    def calculate_severity(self, areas, depths):
        severities = []

        for area, depth in zip(areas, depths):
            max_area_final = 2.0
            min_area_final = 0.1
            estimated_area = area

            ##### calculate the normalized area => [0, 1]
            if (estimated_area >= max_area_final): # in cases that the area score is greater than 2.0 than just assign the normalized value to 1.0
                area_norm = 1.0
            else:
                area_norm = (estimated_area - min_area_final) / (max_area_final - min_area_final)

            total_score = depth + area_norm

            # Categorization of the potholes based on the normalized area and depth values
            if 1.6 <= total_score <= 2.0:
                category = "Critical"
            elif 1.0 <= total_score < 1.6:
                category = "High"
            elif 0.6 <= total_score < 1.0:
                category = "Moderate"
            elif 0.0 <= total_score < 0.6:
                category = "Low"
            else: # for potholes 'not on the road' => in that case total_score = -1
                category = "NA"

            severities.append(category)
        return severities
            

    def run(self):
        """Calculate severity based on area and depth"""
        pothole_areas = self.estimate_area()
        pothole_depths = self.estimate_depth(self.image, pothole_areas)
        # severities = self.calculate_severity(pothole_areas, pothole_depths)

        #TODO: NATHAN make API endpoint call to send severity results

        # TODO: NATHAN DEBUG CODE
        for i, detection in enumerate(self.detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            with open("severity_results.txt", "a") as f:
                f.write(f"Detection {i+1}: BBox={bbox}, Confidence={confidence:.2f}, Area={pothole_areas[i]:.4f} m^2, Depth={pothole_depths[i]:.4f} m\n")
            print(f"Detection {i+1}: BBox={bbox}, Confidence={confidence:.2f}, Area={pothole_areas[i]:.4f} m^2, Depth={pothole_depths[i]:.4f} m")
        return pothole_areas, pothole_depths
    
class FilteringService():
    def __init__(self, image, detections):
        self.image = image
        self.detections = detections
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # TODO: NATHAN verify its running on GPU
        self.num_classes = 19
        self.decode_fn = Cityscapes.decode_target
        self._init_model()
        
        # Initialize transformations
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def _init_model(self):
        # Create the DeepLabV3+ model
        self.model = network.modeling.__dict__[self.config.DEEPLAB_MODEL](
            num_classes=self.num_classes, 
            output_stride=self.config.OUTPUT_STRIDE
        )
        utils.set_bn_momentum(self.model.backbone, momentum=0.01)
        
        # load checkpoint => using CITYSCAPES WEIGHTS for road segmentation
        if os.path.isfile(self.config.DEEPLAB_CHECKPOINT_FILE):
            checkpoint = torch.load(
                self.config.DEEPLAB_CHECKPOINT_FILE, 
                map_location=torch.device('cpu'),
                weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state"])
            print(f"Loaded segmentation model from {self.config.DEEPLAB_CHECKPOINT_FILE}")
            del checkpoint
        else:
            print("[!] Warning: No checkpoint found for segmentation model")
        
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()

    def segment_image(self, image):
        with torch.no_grad():
            img_tensor = self.transform(image).unsqueeze(0)
            
            outputs = self.model(img_tensor)
            predictions = outputs.max(1)[1].cpu().numpy()[0] # TODO: NATHAN verify if this can use GPU instead of CPU

            # Create road mask (class index 1 is road in Cityscapes but 0 is the one for road??)
            road_mask = (predictions == 0).astype(np.uint8)
            return road_mask, predictions
        
    def filter_detections(self, road_mask, full_seg):
        filtered_values = []
        min_road_threshold = 0.60  # Minimum percentage of pothole pixels that must be on the road to be considered valid
        for detection in self.detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            x1, y1, x2, y2 = map(int, bbox)
            is_on_road = False
            total_num_points = 0
            num_points_on_road = 0
            num_points_on_vegetation = 0
            step = 5
            for x in range(x1, x2, step):
                for y in range(y1, y2, step):
                    x = x-1
                    y = y-1
                    if 0 <= x <= road_mask.shape[1] and 0 <= y <= road_mask.shape[0]:
                        total_num_points += 1
                        if road_mask[y, x] == 1:
                            num_points_on_road += 1
                        elif full_seg[y, x] == 8:
                            num_points_on_vegetation += 1 

            percentage_pixels_on_road = (num_points_on_road / total_num_points)
            percentage_pixels_on_vegetation = (num_points_on_vegetation / total_num_points)

            if percentage_pixels_on_road >= min_road_threshold:
                is_on_road = True
            elif confidence >= 0.80 and percentage_pixels_on_road >= 0.30: # TODO: NATHAN update maybe this elif check not needed
                is_on_road = True
            elif confidence >= 0.75 and percentage_pixels_on_vegetation >= 0.5 and percentage_pixels_on_road >= 0.05:
                is_on_road = True
            filtered_values.append(is_on_road)
        return filtered_values
    
    def run(self):
        road_mask, full_seg = self.segment_image(self.image)
        filtered_values = self.filter_detections(road_mask, full_seg)
        # TODO: NATHAN DEBUG CODE
        for i, detection in enumerate(self.detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            on_road = filtered_values[i]
            with open("segmentation_results.txt", "a") as f:
                f.write(f"Detection {i+1}: BBox={bbox}, Confidence={confidence:.2f}, On Road={on_road}\n")
        return filtered_values
            

if __name__ == '__main__':
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor
    
    detection_queue = queue.Queue()

    # Main worker function to create 2 parallel threads for 1) severity calculation and 2) segmentation/filtering
    def process_detection(frame, detections):
        areas, depths, filtered_values = None, None, None

        def severity_task():
            nonlocal areas, depths
            severityService = SeverityCalculationService(frame, detections)
            areas, depths = severityService.run()

        def filtering_task():
            nonlocal filtered_values
            filteringService = FilteringService(frame, detections)
            filtered_values = filteringService.run()
        try:
            t1 = threading.Thread(target=severity_task)
            t2 = threading.Thread(target=filtering_task)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            return areas, depths, filtered_values
        except Exception as e:
            print(f"Error processing detection: {e}")
            return None, None, None


    # Process 1: Pothole detection from webRTC stream from CSI camera on raspi
    potholeDetectionService = PotholeDetectionService()
    detection_thread = threading.Thread(target=potholeDetectionService.run, daemon=True)
    detection_thread.start()

    
    # Process 2-6: Calls a maximum of 5 workers to perform process detection function
    #              This function spins off 2 parallel threads for severity calculation and segmentation/filtering
    with ThreadPoolExecutor(max_workers=5) as executor:
        while True:
            # Check for new detected pothole frames
            if potholeDetectionService.new_pothole_detection:
                frame = potholeDetectionService.latest_detection_frame
                detections = potholeDetectionService.latest_detections
                
                detection_queue.put((frame, detections))
                potholeDetectionService.new_pothole_detection = False
                
                print(f"Added detection to queue. Queue size: {detection_queue.qsize()}")
            
            # Process items from queue using thread pool
            if not detection_queue.empty():
                frame, detections = detection_queue.get()
                
                future = executor.submit(process_detection, frame, detections)
                
                results = future.result()
                areas, depths, filtered_values = results

                # TODO: NATHAN make API call to send results per detection ID
                
            time.sleep(0.1)  # Avoid busy waiting
