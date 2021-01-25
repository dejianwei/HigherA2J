import pyrealsense2 as rs
import cv2

class Realsense():
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)

    def get_intrinsics(self):
        img = self.get_depthimg()
        depth_prof=img.get_profile().as_video_stream_profile()
        i=depth_prof.get_intrinsics()
        principal_point = (i.ppx, i.ppy)
        focal_length = (i.fx, i.fy)
        model = i.model
        return principal_point, focal_length, model

    def get_depthimg(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        while not depth_frame:
            depth_frame = frames.get_depth_frame()
        return depth_frame

    def get_jetcolor_depthimg(self, depthimg):
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depthimg, alpha=0.03), cv2.COLORMAP_JET)
        return depth_colormap

    def stop(self):
        self.pipeline.stop()

if __name__ == '__main__':
    sense = Realsense()
    principal_point, focal_length, model = sense.get_intrinsics()

    print(principal_point)
    print(focal_length)
    print(model)
