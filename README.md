# EdgeFX - Advanced Edge Detection Plugin for GTA San Andreas

A high-quality post-processing plugin for GTA San Andreas that provides advanced edge detection with dual algorithms (Sobel and Canny). All parameters are normalized for GTA SA's coordinate system.

## Features

### Edge Detection
- **Dual Methods**: Choose between Sobel and Canny edge detection algorithms
- **Sobel Operator**: Fast, efficient edge detection for real-time applications
- **Canny Algorithm**: High-quality edge detection with noise reduction and double thresholding
- **Customizable Threshold**: Adjust sensitivity for edge detection (0.05-0.3, normalized for GTA SA scale)
- **Variable Strength**: Control edge visibility intensity (0.1-3.0, optimized for game visibility)
- **Thickness Control**: Adjust edge line thickness (0.1-5.0, optimized for game scale)
- **Custom Colors**: Full RGBA color control for edges with proper alpha blending
- **Color Cycling**: Animated rainbow color cycling on edges (adjusted for game time scale)
- **Edge Fading**: Multiple fading modes with distance-based, screen-edge, and gradient options



## Technical Details

- **Shader Model**: 3.0 (DirectX 9 compatible)
- **Post-Processing**: Full-screen quad rendering
- **Performance**: Optimized for real-time rendering
- **Compatibility**: Works with GTA San Andreas Plugin SDK
- **Configuration**: INI file with comprehensive settings
- **Texel Size**: Optimized for game scale (0.0015 normalized)

## Edge Detection Methods

### Sobel Method (Default)
- **Speed**: Fast and efficient
- **Accuracy**: Good for real-time applications
- **Noise**: May produce some noise
- **Best For**: Lower-end systems, real-time applications
- **Parameters**: `EdgeThreshold` (0.05-0.3 range, normalized for GTA SA scale)
- **Implementation**: Based on exact Sobel filter tutorial source code
- **Kernels**: Gx = [[1,0,-1],[2,0,-2],[1,0,-1]], Gy = [[1,2,1],[0,0,0],[-1,-2,-1]]
- **Grayscale**: Uses exact weights (0.2126, 0.7152, 0.0722) from tutorial

### Canny Method (Advanced)
- **Speed**: Medium performance impact
- **Accuracy**: More accurate edge detection
- **Noise**: Includes noise reduction (Gaussian blur)
- **Features**: Double thresholding for better edge quality with separate alpha control
- **Best For**: Higher-end systems, quality-focused applications
- **Parameters**: 
  - `CannyLowThreshold` (0.05-0.3 range, adjusted for game scale)
  - `CannyHighThreshold` (0.1-0.5 range, optimized for game objects)
  - `CannyGaussianSigma` (0.8-2.5 range, balanced for game detail)
  - `CannyLowAlpha` (0.0-1.0 range, alpha for weak edges)
  - `CannyHighAlpha` (0.0-1.0 range, alpha for strong edges)
  - `CannyLowFadingStrength` (0.0-1.0 range, fading strength for weak edges)
  - `CannyHighFadingStrength` (0.0-1.0 range, fading strength for strong edges)
- **Implementation**: Uses exact Sobel kernels from tutorial for gradient calculation
- **Grayscale**: Uses exact weights (0.2126, 0.7152, 0.0722) from tutorial
- **Alpha Control**: Separate transparency control for weak vs strong edges
- **Normalization**: All parameters optimized for GTA SA coordinate system

## Effect Order

1. **Edge Detection** → Highlights detected edges with custom colors
2. **Edge Fading** → Applies distance-based or screen-edge fading to edges

## Bloom Compatibility

The plugin includes special compatibility features for working with bloom graphics mods:

- **Bloom Compatibility Mode**: Automatically adjusts render states to work with bloom effects
- **Multiple Blending Modes**: 9 different blending options for various visual effects
- **Render State Preservation**: Saves and restores render states to prevent conflicts with other post-processing effects
- **Oversaturation Protection**: Clamps final colors to prevent oversaturation when combined with bloom

**Recommended Settings for Bloom Mods:**
- `BloomCompatibilityMode = true`
- `BlendingMode = 1` (Bloom-compatible blending)
- Adjust `EdgeStrength` and `EdgeColorA` if edges appear too bright or too faint
- For black edges, the bloom-compatible mode uses darkening instead of additive blending

**Blending Mode Guide:**
- **Mode 0 (Normal)**: Standard alpha blending - works well for most cases
- **Mode 1 (Bloom-Compatible)**: Special handling for black edges with bloom mods
- **Mode 2 (Multiply)**: Darkens the image - good for dark edges
- **Mode 3 (Screen)**: Lightens the image - good for bright edges
- **Mode 4 (Overlay)**: Enhances contrast - good for dramatic effects

## Installation

1. Place the compiled DLL in your GTA San Andreas directory
2. Configure settings in `EdgeFX.ini`
3. Launch the game

## Configuration

The plugin creates an `EdgeFX.ini` file with comprehensive settings for all features. See the INI file for detailed parameter descriptions and recommended values.

## Performance Notes

- **Sobel Edge Detection**: Low performance impact (~5-10%)
- **Canny Edge Detection**: Medium performance impact (~15-25%)
- **Color Cycling**: Minimal performance impact
- **Edge Fading**: Minimal performance impact (~1-2%)
- **All parameters optimized for GTA SA coordinate system**

## Troubleshooting

- If edges are too thick: Reduce `EdgeThickness`
- If edges are too thin: Increase `EdgeThickness`
- If too many edges: Increase `EdgeThreshold` (Sobel) or `CannyHighThreshold` (Canny)
- If too few edges: Decrease `EdgeThreshold` (Sobel) or `CannyLowThreshold` (Canny)
- If edges are too strong: Reduce `EdgeStrength`
- If edges are too weak: Increase `EdgeStrength`
- If Canny edges are too noisy: Increase `CannyGaussianSigma`
- If Canny edges are too blurry: Decrease `CannyGaussianSigma`
- If Canny detects too many weak edges: Increase `CannyLowThreshold`
- If Canny misses important edges: Decrease `CannyHighThreshold`
- If weak Canny edges are too visible: Reduce `CannyLowAlpha`
- If weak Canny edges are too faint: Increase `CannyLowAlpha`
- If strong Canny edges are too visible: Reduce `CannyHighAlpha`
- If strong Canny edges are too faint: Increase `CannyHighAlpha`
- If edge fading is too strong: Reduce `EdgeFadingStrength`
- If edge fading is too weak: Increase `EdgeFadingStrength`
- If edge fading distance is too short: Increase `EdgeFadingDistance`
- If edge fading distance is too long: Decrease `EdgeFadingDistance`
- If weak Canny edges fade too much: Increase `CannyLowFadingStrength`
- If weak Canny edges don't fade enough: Decrease `CannyLowFadingStrength`
- If strong Canny edges fade too much: Increase `CannyHighFadingStrength`
- If strong Canny edges don't fade enough: Decrease `CannyHighFadingStrength`
- If edges conflict with bloom effects: Enable `BloomCompatibilityMode` and set `BlendingMode = 1`
- If edges are too bright with bloom: Reduce `EdgeStrength` or `EdgeColorA`
- If edges disappear with bloom: Increase `EdgeStrength` or `EdgeColorA`
- If edges are too dark: Try `BlendingMode = 3` (Screen)
- If edges are too light: Try `BlendingMode = 2` (Multiply)
- If edges need more contrast: Try `BlendingMode = 4` (Overlay) 
