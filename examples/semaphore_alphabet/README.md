# Semaphore alphabet with MoveNet on DepthAI

This demo demonstrates the recognition of semaphore alphabet.

![Semaphore alphabet](medias/semaphore.gif)

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

## Credits

* [Semaphore with The RCR Museum](https://www.youtube.com/watch?v=DezaTjQYPh0&ab_channel=TheRoyalCanadianRegimentMuseum)