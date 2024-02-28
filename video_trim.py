import os
from moviepy.editor import VideoFileClip

def trim_videos(input_folder, output_folder, target_duration):
    """
    Trims videos in the input_folder that are longer than target_duration (in minutes)
    and saves the middle target_duration minutes of the video to the output_folder.
    """
    # Convert target_duration from minutes to seconds
    target_duration_sec = target_duration * 60

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            filepath = os.path.join(input_folder, filename)
            try:
                print(f"Beginning to process {filename}")
                with VideoFileClip(filepath) as clip:
                    duration = clip.duration

                    # Check if the video is longer than the target duration
                    if duration > target_duration_sec:
                        start_trim = (duration - target_duration_sec) / 2
                        end_trim = start_trim + target_duration_sec
                        trimmed_clip = clip.subclip(start_trim, end_trim)

                        # Save the trimmed video
                        output_filepath = os.path.join(output_folder, filename)
                        trimmed_clip.write_videofile(output_filepath, codec="libx264")

                        print(f"Trimmed {filename} to {target_duration} minutes.")
            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")
            finally:
                # This block runs whether or not an exception occurred
                print(f"Finished processing {filename}.")

def main():

    # Path to the script
    script_path = os.path.abspath(__file__)
    # Directory of the script
    script_dir = os.path.dirname(script_path)
    # Path to the raw_videos directory
    input_folder = os.path.join(script_dir, 'raw_videos')
    # Path to the trimmed_videos directory
    output_folder = os.path.join(script_dir, 'trimmed_videos')
    target_duration = 3 # minutes

    trim_videos(input_folder, output_folder, target_duration)

if __name__ == "__main__":
    main()
