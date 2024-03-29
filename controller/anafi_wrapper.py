import math
import threading
import time
from olympe import Drone
import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, moveBy
from olympe.messages.ardrone3.Piloting import PCMD, moveTo, moveBy
from olympe.messages.rth import set_custom_location, return_to_home
from olympe.messages.ardrone3.PilotingState import moveToChanged
from olympe.messages.common.CommonState import BatteryStateChanged
from olympe.messages.ardrone3.PilotingState import AttitudeChanged, GpsLocationChanged, AltitudeChanged, FlyingStateChanged
from olympe.messages.ardrone3.GPSState import NumberOfSatelliteChanged
from olympe.messages.gimbal import set_target, attitude
from olympe.messages.wifi import rssi_changed
from olympe.messages.battery import capacity
from olympe.messages.common.CalibrationState import MagnetoCalibrationRequiredState
import olympe.enums.move as move_mode
import olympe.enums.gimbal as gimbal_mode

from .abs.drone_wrapper import DroneWrapper

def cap_distance(distance):
    if distance < 20:
        return 20
    elif distance > 100:
        return 100
    return distance

class TelloWrapper(DroneWrapper):
    def __init__(self):
        # use the sim
        self.ip = '10.202.0.1'
        self.drone = Drone(self.ip)
        self.active = False
        
        self.lowdelay = False
        
        # self.drone = Tello()
        # self.active_count = 0
        # self.stream_on = False

    def keep_active(self):
        pass

    def connect(self):
        self.drone.connect()
        self.active = True

    def takeoff(self) -> bool:
        
        if not self.is_battery_good():
            return False
        
        self.drone(TakeOff())
        self.hovering()
        
    def hover(self):
        self.PCMD(0, 0, 0, 0)

    def land(self):
        self.drone(Landing()).wait().success()

    def start_stream(self):
        if self.lowdelay:
            self.streamingThread = LowDelayStreamingThread(self.drone, self.ip)
        else:
            self.streamingThread = StreamingThread(self.drone, self.ip)
        self.streamingThread.start()

    def stop_stream(self):
        self.streamingThread.stop()

    def get_frame_reader(self):
         if self.streamingThread:
            return self.streamingThread.grabFrame()

    def move_forward(self, distance: int) -> bool:
        capdistance = cap_distance(distance)
        self.drone(moveBy(capdistance, 0, 0, 0)).wait()
        time.sleep(0.5)
        return True

    def move_backward(self, distance: int) -> bool:
        capdistance = cap_distance(distance)
        self.drone(moveBy(-capdistance, 0, 0, 0)).wait()
        time.sleep(0.5)
        return True

    def move_left(self, distance: int) -> bool:
        capdistance = cap_distance(distance)
        self.drone(moveBy(0, -capdistance, 0, 0)).wait()
        time.sleep(0.5)
        return True

    def move_right(self, distance: int) -> bool:
        capdistance = cap_distance(distance)
        self.drone(moveBy(0, capdistance, 0, 0)).wait()
        time.sleep(0.5)
        return True

    def move_up(self, distance: int) -> bool:
        capdistance = cap_distance(distance)
        self.drone(moveBy(0, 0, -capdistance, 0)).wait()
        time.sleep(0.5)
        return True

    def move_down(self, distance: int) -> bool:
        capdistance = cap_distance(distance)
        self.drone(moveBy(0, 0, capdistance, 0)).wait()
        time.sleep(0.5)
        return True

    def turn_ccw(self, degree: int) -> bool:
        # Convert degrees to radians for Olympe
        radians = math.radians(degree)
        self.drone(moveBy(0, 0, 0, -radians)).wait()
        time.sleep(2.5)
        return True

    def turn_cw(self, degree: int) -> bool:
        # Convert degrees to radians for Olympe
        radians = math.radians(degree)
        self.drone(moveBy(0, 0, 0, radians)).wait()
        time.sleep(2.5)
        return True
    
    def is_battery_good(self):
        self.battery = self.drone.get_state(BatteryStateChanged)["percent"]
        print(f"> Battery level: {self.battery}% ", end='')
        if self.battery < 20:
            print('is too low [WARNING]')
        else:
            print('[OK]')
            return True
        return False




import cv2
import numpy as np
import os

class StreamingThread(threading.Thread):

    def __init__(self, drone, ip):
        threading.Thread.__init__(self)
        self.currentFrame = None 
        self.drone = drone
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        self.cap = cv2.VideoCapture(f"rtsp://{ip}/live", cv2.CAP_FFMPEG)
        self.isRunning = True

    def run(self):
        try:
            while(self.isRunning):
                ret, self.currentFrame = self.cap.read()
        except Exception as e:
            print(e)

    def grabFrame(self):
        try:
            frame = self.currentFrame.copy()
            return frame
        except Exception as e:
            # Send a blank frame
            return np.zeros((720, 1280, 3), np.uint8) 

    def stop(self):
        self.isRunning = False

import queue

class LowDelayStreamingThread(threading.Thread):

    def __init__(self, drone, ip):
        threading.Thread.__init__(self)
        self.drone = drone
        self.frame_queue = queue.Queue()
        self.currentFrame = np.zeros((720, 1280, 3), np.uint8)

        self.drone.streaming.set_callbacks(
            raw_cb=self.yuvFrameCb,
            h264_cb=self.h264FrameCb,
            start_cb=self.startCb,
            end_cb=self.endCb,
            flush_raw_cb=self.flushCb,
        )

    def run(self):
        self.isRunning = True

        while self.isRunning:
            try:
                yuv_frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self.copyFrame(yuv_frame)
            yuv_frame.unref()
    
    def grabFrame(self):
        try:
            frame = self.currentFrame.copy()
            return frame
        except Exception as e:
            # Send a blank frame
            return np.zeros((720, 1280, 3), np.uint8) 

    def copyFrame(self, yuv_frame):
        info = yuv_frame.info()

        height, width = (  # noqa
            info["raw"]["frame"]["info"]["height"],
            info["raw"]["frame"]["info"]["width"],
        )
        
        cv2_cvt_color_flag = {
            olympe.VDEF_I420: cv2.COLOR_YUV2BGR_I420,
            olympe.VDEF_NV12: cv2.COLOR_YUV2BGR_NV12,
        }[yuv_frame.format()]

        self.currentFrame = cv2.cvtColor(yuv_frame.as_ndarray(), cv2_cvt_color_flag)

    ''' Callbacks '''

    def yuvFrameCb(self, yuv_frame):
        """
        This function will be called by Olympe for each decoded YUV frame.

            :type yuv_frame: olympe.VideoFrame
        """
        yuv_frame.ref()
        self.frame_queue.put_nowait(yuv_frame)

    def flushCb(self, stream):
        if stream["vdef_format"] != olympe.VDEF_I420:
            return True
        while not self.frame_queue.empty():
            self.frame_queue.get_nowait().unref()
        return True

    def startCb(self):
        pass

    def endCb(self):
        pass

    def h264FrameCb(self, h264_frame):
        pass

    def stop(self):
        self.isRunning = False
        # Properly stop the video stream and disconnect
        assert self.drone.streaming.stop()