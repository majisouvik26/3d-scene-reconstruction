import os
import cv2
import glob

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

if __name__ == "__main__":
    input_folder = "output/reconstructed_views_truck"
    output_video = "output/truck_reconstruction.mp4"
    
    # Create the video with 30 fps
    create_video(input_folder, output_video, fps=30)
