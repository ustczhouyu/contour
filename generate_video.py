import cv2
import os
import tqdm

image_folder = '/home/sumche/datasets/cityscapes/demoVideo/stuttgart_00'
video_name = '/home/sumche/datasets/cityscapes/demoVideo/stuttgart_00.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'MPEG')

video = cv2.VideoWriter(video_name, fourcc, 30.0, (width, height))

for image in tqdm.tqdm(images):
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
