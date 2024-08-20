from pytube import YouTube
from moviepy.editor import AudioFileClip
import cv2
import os
from PIL import Image
import imagehash
import argparse
import numpy as np
import platform
import subprocess
import zipfile
import webbrowser
import os
import re
from numpy.fft import fftshift, fft2


def download_youtube_video(url, save_path="", audio_only=False):
    print(f"Downloading {'audio' if audio_only else 'video'} from {url} ...")
    yt = YouTube(url)
    if audio_only:
        stream = yt.streams.get_audio_only()
    else:
        stream = (
            yt.streams.filter(progressive=True, file_extension="mp4")
            .order_by("resolution")
            .desc()
            .first()
        )
    if not save_path:
        save_path = stream.default_filename

    stream.download(output_path=save_path)
    print(f"{'Audio' if audio_only else 'Video'} downloaded: {stream.default_filename}")

    return os.path.join(save_path, stream.default_filename)


def is_mostly_black_or_white(image, threshold, white_threshold=225):
    """
    Check if the given image is mostly black or white.
    :param image: Image to be checked.
    :param threshold: Threshold below which the image is considered 'black'.
    :param white_threshold: Threshold above which the image is considered 'white'.
    :return: True if the image is mostly black or white; False otherwise.
    """
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the average brightness of the image
    average_brightness = cv2.mean(gray_image)[0]

    return average_brightness < threshold or average_brightness > white_threshold


def extract_frames(
    video_path,
    frame_interval=50,
    save_path="",
    black_white_threshold=10,
    hash_func=imagehash.average_hash,
    hash_size=8,
    hash_diff_threshold=10,
    remove_blur=True,
    motion_blur_threshold=-0.02,
):
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {total_frames}, FPS: {fps}")

    frame_index = 0
    saved_frame_count = 0
    previous_hash = None

    while True:
        ret, frame = cap.read()
        if ret:
            if frame_index % frame_interval == 0:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                is_black_or_white = is_mostly_black_or_white(
                    frame, threshold=black_white_threshold
                )
                current_hash = hash_func(pil_image, hash_size=hash_size)

                is_blurry, blur_score = detect_motion_blur(frame, motion_blur_threshold)

                should_save = (
                    True  # Initialize the flag assuming the frame will be saved
                )

                if previous_hash is not None:
                    hash_diff = current_hash - previous_hash
                    if hash_diff < hash_diff_threshold:
                        should_save = False  # If frames are similar, do not save
                        print(
                            f"Skipping frame {frame_index} due to perceived duplication (hash diff: {hash_diff})"
                        )

                if is_black_or_white:
                    should_save = (
                        False  # If frame is mostly black or white, do not save
                    )
                    print(
                        f"Skipping frame {frame_index} because it's mostly black or white"
                    )

                if remove_blur and is_blurry:
                    should_save = False
                    print(
                        f"Skipping frame {frame_index} because it's blurry (blur score: {blur_score}))"
                    )

                if should_save:
                    save_filename = os.path.join(save_path, f"frame_{frame_index}.jpg")
                    cv2.imwrite(save_filename, frame)
                    saved_frame_count += 1

                previous_hash = current_hash

            frame_index += 1
        else:
            break

    cap.release()
    print(
        f"Done extracting frames. {saved_frame_count} images are saved in '{save_path}'."
    )


def open_file_explorer(path):
    """
    Open the file explorer at the specified path.
    """
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.Popen(["open", path])
    else:  # linux
        subprocess.Popen(["xdg-open", path])


def user_image_confirmation(path):
    """
    Prompt the user to check the images before proceeding.
    """
    confirmation = input("Press Enter to open finder to check the images.")
    open_file_explorer(
        path
    )  # This will open the file explorer so you can check the images

    confirmation = input(
        "Have you checked the images and do you want to proceed with posting a training? (y/n): "
    )
    return confirmation.lower() == "y"


def user_audio_confirmation(path):
    """
    Prompt the user to check the audio before proceeding.
    """
    confirmation = input("Press Enter to check the audio file")
    open_file_explorer(path)

    confirmation = input(
        "Have you checked the audio and do you want to proceed with posting a training? (y/n): "
    )
    return confirmation.lower() == "y"


def user_model():
    """
    Prompt the user to input the fine tune model name
    """
    does_model_exist = input("Have you already created the model on Replicate? (y/n): ")

    if does_model_exist.lower() == "y":
        model_name = input("Please input the model name (owner/model_name): ")
        return model_name
    else:
        name = input(
            "What do you want to call the model? Pick a short and memorable name. Use lowercase characters and dashes. (eg: sdxl-barbie, musicgen-ye): "
        )
        webbrowser.open(f"https://replicate.com/create?name={name}")
        input(
            "Once you have created the model (click Create on the webpage that just opened), press Enter to continue."
        )
        owner = input("What is your Replicate username? ")
        return f"{owner}/{name}"


def zip_directory(folder_path, zip_path):
    """
    Compress a directory (with all files in it) into a zip file.

    :param folder_path: Path of the folder you want to compress.
    :param zip_path: Destination file path, including the filename of the new zip file.
    """
    print(f"Zipping {folder_path} to {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))

    return zip_path


def create_sdxl_training(model, save_dir, caption_prefix="in the style of TOK"):
    try:
        # Please make sure that 'replicate' is installed and available in your system's PATH.
        # The command assumes that "nightmare.zip" is correctly placed and accessible.
        command = [
            "replicate",
            "train",
            "stability-ai/sdxl",
            "--destination",
            model,
            "--web",
            f"input_images=@{save_dir}",
            f"caption_prefix={caption_prefix}",
        ]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error executing the command:", str(e))
    except FileNotFoundError:
        print("Error: 'replicate' command not found. Is it installed correctly?")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def create_musicgen_training(model, save_dir, audio_description, drop_vocals=False):
    try:
        # Please make sure that 'replicate' is installed and available in your system's PATH.
        # The command assumes that "your-audio.mp3" is correctly placed and accessible.
        command = [
            "replicate",
            "train",
            "sakemin/musicgen-fine-tuner",
            "--destination",
            model,
            "--web",
            "model=medium",
            f"drop_vocals={drop_vocals}",
            f"one_same_description={audio_description}",
            f"dataset_path=@{save_dir}",
        ]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error executing the command:", str(e))
    except FileNotFoundError:
        print("Error: 'replicate' command not found. Is it installed correctly?")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def is_replicate_api_token_set():
    return "REPLICATE_API_TOKEN" in os.environ


def is_replicate_cli_installed():
    try:
        subprocess.check_output(["replicate", "--version"])
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False


def slugify(title):
    """
    Slugify a YouTube title.

    :param title: The title to slugify.
    :return: The slugified title.
    """
    return re.sub(r"\W+", "-", title).lower()


def detect_motion_blur(image, motion_blur_threshold):
    # Load the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Scale the image to the range [0, 1]
    image = image / 255.0

    # Apply DCT
    dct = cv2.dct(np.float32(image))

    # Compute the average of the DCT coefficients in the high-frequency region
    avg = np.mean(dct[1 : int(dct.shape[0] / 2), 1 : int(dct.shape[1] / 2)])

    avg = avg * 10000 if avg else 0

    # If the average is below a certain threshold, the image is likely blurred
    # After some trial and error, around -0.02 seems to be a good threshold
    if avg < motion_blur_threshold:
        return (True, avg)
    else:
        return (False, avg)


def convert_mp4_to_mp3(mp4_file_path):
    print(f"Converting {mp4_file_path} to MP3 ...")
    audio_clip = AudioFileClip(mp4_file_path)
    mp3_file_path = mp4_file_path.replace(".mp4", ".mp3")
    audio_clip.write_audiofile(mp3_file_path, codec="mp3")
    audio_clip.close()
    os.remove(mp4_file_path)
    return mp3_file_path


def process_audio(audio_file_path):
    audio_file_path = convert_mp4_to_mp3(audio_file_path)

    if user_audio_confirmation(audio_file_path):
        # If the user confirms, proceed with the posting function
        model = user_model()

        print(
            "Please describe the audio, use 2 to 3 comma separated keywords. This could be a band name, genre or something unique. You’ll use this when prompting your fine-tune."
        )
        audio_description = input("Describe the audio: ")

        # ask user if they want to drop vocals
        print("MusicGen does not train well with audio that has vocals")
        drop_vocals_input = input(
            "Do you want to automatically drop vocals from your audio? (y/n): "
        )
        drop_vocals = True if drop_vocals_input.lower() == "y" else False

        create_musicgen_training(model, audio_file_path, audio_description, drop_vocals)
    else:
        print("Operation cancelled by the user.")


def process_video(video_file_path, interval, caption_prefix):
    # slugify the video title
    video_name = video_file_path.split("/")[-1]
    output_directory = f"./extracted_frames/{slugify(video_name)}"

    extract_frames(video_file_path, frame_interval=interval, save_path=output_directory)

    # After extracting and saving images, ask the user to confirm
    if user_image_confirmation(output_directory):
        # If the user confirms, proceed with the posting function
        model = user_model()

        # Compress the directory with the images
        zip_path = zip_directory(output_directory, output_directory + ".zip")

        create_sdxl_training(model, zip_path, caption_prefix=caption_prefix)
    else:
        print("Operation cancelled by the user.")


def main():
    parser = argparse.ArgumentParser(
        description="Download a video from YouTube and extract frames or audio"
    )
    parser.add_argument("url", help="URL of the YouTube video")
    parser.add_argument(
        "--interval", help="Interval between frames", default=50, type=int
    )
    parser.add_argument(
        "--caption_prefix",
        help="automatically add this to the start of each caption",
        default="in the style of TOK",
        type=str,
    )
    parser.add_argument("--audio", help="Download audio only", action="store_true")
    parser.add_argument(
        "--remove_blur", help="remove blurry frames", default=True, action="store_true"
    )
    args = parser.parse_args()

    if not is_replicate_cli_installed():
        input(
            "🚫 Replicate CLI is not installed. Please install it before proceeding. Link: https://github.com/replicate/cli. Press any key to open the webpage."
        )
        webbrowser.open(f"https://github.com/replicate/cli")
    else:
        print("✅ Replicate CLI is installed. Proceeding...")

    if not is_replicate_api_token_set():
        print(
            "🚫 REPLICATE_API_TOKEN is not set. Please set it with `export REPLICATE_API_TOKEN=<your-token>`, then try again."
        )
        return
    else:
        print("✅ REPLICATE_API_TOKEN is set. Proceeding...")

    if args.audio:
        print("🎵 Audio training mode. Proceeding...")
    else:
        print("🎥 Video training mode. Proceeding...")

    video_url = args.url
    interval = args.interval
    caption_prefix = args.caption_prefix

    # Directory where you want to save the downloaded video
    download_directory = "./downloaded_audio" if args.audio else "./downloaded_videos"

    if video_url.startswith("http"):
        video_file_path = download_youtube_video(
            video_url, save_path=download_directory, audio_only=args.audio
        )
    else:
        video_file_path = video_url

    if args.audio:
        process_audio(video_file_path)
    else:
        process_video(video_file_path, interval, caption_prefix)


if __name__ == "__main__":
    main()
