# multiresolutionblend
Implementing multiresolution blending using gaussian and laplacian stacks to combine separate images. 

![ScreenShot](http://i.imgur.com/Hv1Lxmx.jpg)

The overall goal of this project was to implement the multiresolution blending method proposed in the MIT paper by Burt and Adelson (http://persci.mit.edu/pub_pdfs/spline83.pdf). This would allow to blend two images seamlessly, without looking like the images were combined haphazardly. We can see an example, as seen below:

## Running the Code
```
python main.py <image1> <image2> <mask>
```

## Running on Example Images
```
python main.py examples/apple.jpg examples/orange.jpg examples/mask.jpg
```
