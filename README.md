## Advanced Lane Lines
This is my submission for project 4 of the Udacity Self-Driving Car Nanodegree. I wrote [a page on Medium](https://medium.com/@edvoas/advanced-lane-finding-a4bb8356824d) for this project if you want to know more details.

Eventually, I hope to return to this and improve it. It needs some refactoring and other love.

I didn't include the files that would have come with the actual Udacity project, but you can get them from [their Github project](https://github.com/udacity/CarND-Advanced-Lane-Lines).

You may also need to provide your own images for some of the code in the notebook.

The main.py file accepts either a image or video file. You can also specify `--mask-only` or `--full-debug` when giving a video file. `--mask-only` will output a file with only color thresholding done to it. `--full-debug` will output a directory with every frame from the video along with it's 'detected' version. It will also write a CSV file with all of the polynomials and derivatives we are detecting for graphing, etc.
