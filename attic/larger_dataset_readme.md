I ended up not using these because about 20K images were sufficient for training:


4. Extract training images from a 4K/UHD blu-ray disk from an attached reader:

Set up hardware/firmware as described here: https://forum.makemkv.com/forum/viewtopic.php?f=16&t=19634

Install makemkvcon from: http://makemkv.com/download/

Rip the 4K/UHD blu-ray disk:

```bash
makemkvcon mkv disc:0 all .
```

From here, rename the 4K .MKV video file containing the movie content to something shorter like `avatar.mkv`.


5. Extract training images from a large high-quality 4K video file:

```bash
# Install ffmpeg if it is not installed yet
sudo apt install ffmpeg

# Extract every ~10th frame as random 512x512 .PNG crops under data/video_name/*.png
python video_to_data.py avatar.mkv
```

This will split the video file into 120 second segments and process them in parallel for better CPU utilization.

The video file name is used to tag the data, so it should be a simple and short string.


6. Extract training images from ImageNet Full (Fall 2011 release):

Download the torrent from https://academictorrents.com/details/564a77c1e1119da199ff32622a1609431b9f1c47

```bash
python imagenet_to_data.py ~/imagenet/fall11_whole.tar
```