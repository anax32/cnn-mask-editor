# Mask Editor

simple opencv-powered python app to mask images with segmentations

## usage

the script takes a pattern for images and masks to edit

sadly, we have to do the x-forwarding in the `docker run` command

```bash
docker run \
  -v $(pwd)/data:/data \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  masked \
  --images=/data/images/*.png \
  --masks=/data/masks/*.png
```

when running, a window will appear displaying the image.
+ press `h` for on-screen help
+ hold `CTRL` and move the mouse to add mask
+ hold `CTRL+SHIFT` and move the mouse to remove mask
+ press `x` to clear the mask
+ press `space` to save the mask and move to the next image
+ press `esc` to exit

The image cannot be resized: if it is bigger than your monitor, resize the image or buy a new monitor.

## tips

+ mask editing is extremely frustrating: be sure to take regular breaks with physical movement to get that frustration out
+ when unmasking, press `SHIFT` first and release it last to avoid accidental masking
