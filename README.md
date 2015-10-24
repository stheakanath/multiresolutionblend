# multiresolutionblend
Implementing multiresolution blending using gaussian and laplacian stacks to combine separate images. 

The overall goal of this project was to implement the multiresolution blending method proposed in the MIT paper by Burt and Adelson (http://persci.mit.edu/pub_pdfs/spline83.pdf). This would allow to blend two images seamlessly, without looking like the images were combined haphazardly. We can see an example, as seen below:

## Apple
<img src="/examples/apple.jpg" style="width: 200px;height:200px"/>
![ScreenShot](/examples/apple.jpg =200x200)

## Orange
![ScreenShot](/examples/orange.jpg =200x200)

## Orapple
![ScreenShot](/examples/orapple.png =200x200)