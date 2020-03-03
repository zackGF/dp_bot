
import glob
import os

VIDEO_DIR = os.path.abspath("/home/zsy/mask_rcnn/videos")   #原始视频存放的文件夹
VIDEO_SAVE_DIR = os.path.join(VIDEO_DIR, "out")   #每一帧图片所存放的文件夹



def make_video(outvid, images=None, fps=30, size=None,
               is_color=True, format="FMP4"):
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    #vid.release()
    return vid

images = list(glob.iglob(os.path.join(VIDEO_SAVE_DIR, '*.*')))
# Sort the images by integer index
print(images)
images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))

outvid = os.path.join(VIDEO_DIR, "out.mp4")
make_video(outvid, images, fps=30)
print('make video success')
