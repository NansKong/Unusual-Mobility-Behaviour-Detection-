"""
download_sample_video.py
Downloads a generic test .mp4 video so you can test the pipeline
without having to manually find the original UCF-Crime raw dataset.
"""

import urllib.request
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent.parent
    videos_dir = base_dir / "data" / "ucf_crime" / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = videos_dir / "Fighting001.mp4"
    
    # We will download a reliable, generic test MP4 file
    # This is "For Bigger Blazes" - a short open-source demo video
    test_video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
    
    print(f"Downloading sample MP4 to {output_path}...")
    try:
        urllib.request.urlretrieve(test_video_url, output_path)
        print("Success! You can now run the pipeline command:")
        print(f"python src/pipeline.py --video \"{output_path}\" --out_dir outputs --device cuda")
    except Exception as e:
        print(f"Failed to download test video: {e}")

if __name__ == "__main__":
    main()
