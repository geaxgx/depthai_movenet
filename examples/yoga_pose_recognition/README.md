# Yoga Pose Classification with MoveNet on DepthAI

This demo demonstrates the classification of yoga poses.

![Yoga Pose Classification](medias/yoga_pose.gif)

Recognizes four yoga poses :
- [mountain](https://en.wikipedia.org/wiki/Tadasana)
- [cobra](https://en.wikipedia.org/wiki/Bhujangasana)
- [Triangle 1](https://en.wikipedia.org/wiki/Trikonasana)
- [Downward Dog](https://en.wikipedia.org/wiki/Downward_Dog_Pose)

## Usage

```
-> python3 demo.py -h
usage: demo.py [-h] [-m {lightning,thunder}] [-i INPUT] [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -m {lightning,thunder}, --model {lightning,thunder}
                        Model to use (default=thunder
  -i INPUT, --input INPUT
                        'rgb' or 'rgb_laconic' or path to video/image file to
                        use as input (default: rgb)
  -o OUTPUT, --output OUTPUT
                        Path to output video file

```

## CSV Folder 

The ```fitness_poses_csvs_out_processed_f ``` contains four csv files, each for each yoga pose. It contains information on the sample images of each class. The first column is the image name, and next there are x and y coordinates corresponding to each joint point for that image, nose, left eye, right eye and so on. It contains x and y coordinates for 17 joint keypoints.

## Image credits

- https://www.yogajournal.com/poses/mountain-pose/
- https://rajyogarishikesh.com/cobra-pose-bhujangasana.html
- https://www.naukrinama.com/stressbuster/simple-yoga-asanas-to-stay-fit-and-young/bhujangasana-cobra-pose/
- https://en.wikipedia.org/wiki/Downward_Dog_Pose
- https://www.ekhartyoga.com/resources/yoga-poses/downward-facing-dog-pose
- https://www.verywellfit.com/extended-triangle-pose-utthita-trikonasana-3567129
- https://www.ekhartyoga.com/resources/yoga-poses/extended-triangle-pose

## References

- https://google.github.io/mediapipe/