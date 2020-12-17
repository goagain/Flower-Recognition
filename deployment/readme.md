# Deployment

## System Requirement

tensorflow 2.3+ <br>
opencv-python <br>

## Arguments

-h or --help：Show Help <br>
-i or --input：input file path <br>
-o or --output: output file path, default is output.mp4 <br>
-t or --type: type of media, available values are [img, video], default is img <br>
-v or --version：Show version <br>

## Sample

python ./main.py -i sample.png <br>
python ./main.py -i input.mp4 -o output.mp4 -t video <br>