import os
import cv2
import glob
from PIL import Image

def create_video(image_folder, output_video, fps=30):
    """
    Create an MP4 video from all images in the specified folder.
    
    Args:
        image_folder (str): Path to folder containing images
        output_video (str): Path where the output video will be saved
        fps (int): Frames per second (default: 30)
    """
    # Get all image files from the folder
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    
    if not image_files:
        print(f"No PNG images found in {image_folder}")
        return
    
    # obtain dimensions
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for image_file in image_files:
        frame = cv2.imread(image_file)
        video.write(frame)
    
    video.release()
    
    print(f"Video created successfully at {output_video}")
    print(f"Total frames: {len(image_files)}")
    print(f"FPS: {fps}")

def create_gif(image_folder, output_gif, duration=100):
    """
    Create a GIF animation from all images in the specified folder.
    
    Args:
        image_folder (str): Path to folder containing images
        output_gif (str): Path where the output GIF will be saved
        duration (int): Duration for each frame in milliseconds (default: 100)
    """
    # Get all image files from the folder
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    
    if not image_files:
        print(f"No PNG images found in {image_folder}")
        return
    
    # Load all
    images = []
    for image_file in image_files:
        img = Image.open(image_file)
        images.append(img)
    
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=True
    )
    
    print(f"GIF created successfully at {output_gif}")
    print(f"Total frames: {len(images)}")
    print(f"Frame duration: {duration}ms")

if __name__ == "__main__":
    input_folder = "output/reconstructed_views_truck"
    output_video = "output/truck_reconstruction.mp4"
    output_gif = "output/truck_reconstruction.gif"
    
    # Create the video with 30 fps
    create_video(input_folder, output_video, fps=30)
    
    # Create the GIF with 100ms duration per frame (10 fps)
    create_gif(input_folder, output_gif, duration=100)
